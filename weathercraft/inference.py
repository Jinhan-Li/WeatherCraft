"""Load the released checkpoints and forecast from two observed daily fields."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from weathercraft.build import build_tokenizer, build_dynamics, load_tokenizer_checkpoint
from weathercraft.config import load_config, write_json
from weathercraft.data.daily import field_dataset, expand_members, advance_history
from weathercraft.data.tokens import seasonal_features
from weathercraft.models.dynamics.kv_cache import BlockKVCache, KVRollout


def observed_fields(config, split, days, index):
    if index:
        previous = days[index - 1]
    else:
        previous_split = {"valid": "train", "test": "valid"}.get(split)
        if previous_split is None:
            raise ValueError("The first training day has no in-split preceding observation")
        prior = field_dataset(config, previous_split)
        previous = prior[len(prior) - 1]
    current = days[index]
    if int(current["day"]) - int(previous["day"]) != 1:
        raise ValueError("Observed history must be consecutive")
    return torch.cat((previous["field"], current["field"]), dim=0)


class Forecaster:
    """The only mutable inference state lives inside each rollout, never in weights."""

    def __init__(self, config, device, checkpoint_path=None):
        self.config, self.device = (config, torch.device(device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer_config = load_config(config["tokenizer_config"])
        self.tokenizer = build_tokenizer(tokenizer_config)
        tokenizer_state = load_tokenizer_checkpoint(
            self.tokenizer, config["tokenizer_checkpoint"], tokenizer_config
        )
        del tokenizer_state
        self.model = build_dynamics(config, self.tokenizer)
        self.checkpoint_path = Path(checkpoint_path or config["checkpoint"])
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        self.model.load_state_dict(state["model"], strict=True)
        self.selected_epoch = int(state["epoch"]) + 1
        del state
        self.tokenizer.to(self.device).eval().requires_grad_(False)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode_history(self, fields):
        """[2,C,H,W] normalized observations -> scale tensors [1,2,Q,h,w]."""
        if fields.ndim != 4 or fields.shape[0] != 2:
            raise ValueError("Expected two observations with shape [2,C,H,W]")
        tokens = self.tokenizer.encode(fields[:, None].to(self.device, dtype=torch.float32))
        return tuple((value[None] for value in tokens))

    def rollout(
        self,
        history,
        origin_day,
        leads,
        members=8,
        mode="sample",
        kv_cache=True,
        temperature=1.0,
        top_k=0,
    ):
        """Yield (lead, normalized member fields); each member feeds itself back."""
        if members < 1 or not leads or min(leads) < 1:
            raise ValueError("Positive members and forecast leads are required")
        history = expand_members(history, members)
        engine = BlockKVCache(self.model) if kv_cache else None
        with torch.inference_mode():
            state = KVRollout(engine, history) if engine else None
            for lead in range(1, max(leads) + 1):
                season = (
                    seasonal_features(int(origin_day) + lead)[None]
                    .to(self.device)
                    .expand(members, -1)
                )
                precision = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with precision:
                    if state:
                        prediction = state.step(
                            season, mode=mode, temperature=temperature, top_k=top_k
                        )
                    else:
                        prediction = self.model.generate(
                            history, season, mode=mode, temperature=temperature, top_k=top_k
                        )
                        history = advance_history(history, prediction)
                    fields = (
                        self.tokenizer.decode(prediction)[:, 0].float() if lead in leads else None
                    )
                if fields is not None:
                    if not torch.isfinite(fields).all():
                        raise FloatingPointError(f"Nonfinite forecast at lead {lead}")
                    yield (lead, fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dynamics.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--origin", required=True)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--mode", choices=["greedy", "sample"], default="sample")
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 10, 15])
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    days = field_dataset(config, args.split)
    matches = np.flatnonzero(days.day_ids == np.datetime64(args.origin, "D").astype(np.int64))
    if len(matches) != 1:
        raise ValueError("Origin must be a date in the requested split")
    index = int(matches[0])
    forecaster = Forecaster(config, "cuda" if torch.cuda.is_available() else "cpu", args.checkpoint)
    history = forecaster.encode_history(observed_fields(config, args.split, days, index))
    torch.manual_seed(config["evaluation"]["seed"] + int(days.day_ids[index]))
    fields = {
        f"lead_{lead:02d}": value.cpu().numpy()
        for lead, value in forecaster.rollout(
            history,
            days.day_ids[index],
            args.leads,
            args.members,
            args.mode,
            not args.no_kv_cache,
            config["evaluation"]["temperature"],
            config["evaluation"]["top_k"],
        )
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            **fields,
            variable_names=np.array(days.variable_names),
            latitude=days.latitudes.numpy(),
            longitude=days.base.longitude,
            mean=days.mean.numpy(),
            std=days.std.numpy(),
        )
    write_json(
        output.with_suffix(".json"),
        dict(
            origin=args.origin,
            leads=args.leads,
            members=args.members,
            mode=args.mode,
            units="normalized; physical = field * std + mean",
            kv_cache=not args.no_kv_cache,
        ),
    )
    print(f"FORECAST_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
