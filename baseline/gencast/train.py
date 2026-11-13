"""Single-host eight-device pmap training from scratch."""

import argparse
import json
import pickle
import time
import numpy as np
from .common import CODE, CFG, OUT, Data, epoch_indices, write_json
import jax
import jax.numpy as jnp
import optax
from .model import transforms


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(jax.device_get(obj), f, protocol=5)
    tmp.replace(path)


def replicated(params):
    return jax.device_put_replicated(params, jax.local_devices())


def unrep(tree):
    return jax.tree.map(lambda a: np.array(a[0]), tree)


def finite(tree):
    return all((np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(tree)))


class Engine:

    def __init__(self, data, batch):
        assert jax.local_device_count() == 8 and all(
            (x.platform == "gpu" for x in jax.local_devices())
        )
        self.data = data
        self.batch = batch
        self.accum = 4 // batch
        assert batch in (1, 2, 4)
        self.tr = transforms(data)
        x, y, p = data.batch(data.centers("train")[:1])
        params = jax.jit(self.tr["loss"].init)(jax.random.PRNGKey(CFG["seed"]), x, y, p)
        self.parameter_count = sum((v.size for v in jax.tree.leaves(params)))
        self.params = replicated(params)
        steps = len(data.centers("train")) // 32 * CFG["epochs"]
        self.schedule = optax.warmup_cosine_decay_schedule(
            0.0, CFG["learning_rate"], CFG["warmup_updates"], steps, end_value=0.0
        )
        self.optimizer = optax.adamw(
            self.schedule, b1=0.9, b2=0.999, eps=1e-08, weight_decay=CFG["weight_decay"]
        )
        self.opt = replicated(self.optimizer.init(params))

        def step(params, opt, key, x, y, phase):

            def body(carry, inputs):
                total, g = carry
                k, a, b, c = inputs
                value, grad = jax.value_and_grad(self.tr["loss"].apply)(params, k, a, b, c)
                return ((total + value, jax.tree.map(jnp.add, g, grad)), None)

            zero = jax.tree.map(jnp.zeros_like, params)
            (loss, grad), _ = jax.lax.scan(
                body, (jnp.float32(0), zero), (jax.random.split(key, self.accum), x, y, phase)
            )
            grad = jax.lax.pmean(jax.tree.map(lambda a: a / self.accum, grad), "devices")
            loss = jax.lax.pmean(loss / self.accum, "devices")
            norm = optax.global_norm(grad)
            updates, opt = self.optimizer.update(grad, opt, params)
            params = optax.apply_updates(params, updates)
            return (params, opt, loss, norm)

        self.step = jax.pmap(step, axis_name="devices", donate_argnums=(0, 1))
        self.validate = jax.pmap(self.tr["loss"].apply)
        self.sample = jax.pmap(self.tr["sample"].apply)
        self.denoise = jax.pmap(self.tr["denoise"].apply)
        self.single_loss = jax.jit(self.tr["loss"].apply)

    def update(self, ids, update, seed=42):
        x, y, p = self.data.batch(ids)

        def shape(a):
            return a.reshape((8, self.accum, self.batch) + a.shape[1:])

        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), update), 8)
        self.params, self.opt, loss, grad = self.step(
            self.params, self.opt, keys, *map(shape, (x, y, p))
        )
        loss, grad = (np.asarray(loss), np.asarray(grad))
        if not np.isfinite(loss).all() or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite loss/gradient")
        return (float(loss[0]), float(grad[0]))

    def valid(self):
        ids = self.data.centers("valid")
        total = 0.0
        count = 0
        for begin in range(0, len(ids) // 8 * 8, 8):
            x, y, p = self.data.batch(ids[begin : begin + 8])
            keys = jax.random.split(jax.random.PRNGKey(100000 + begin), 8)
            values = np.asarray(
                self.validate(self.params, keys, x[:, None], y[:, None], p[:, None])
            )
            assert np.isfinite(values).all()
            total += float(values.sum())
            count += 8
        p0 = jax.tree.map(lambda a: a[0], self.params)
        for pos in range(count, len(ids)):
            x, y, p = self.data.batch(ids[pos : pos + 1])
            value = float(self.single_loss(p0, jax.random.PRNGKey(100000 + pos), x, y, p))
            assert np.isfinite(value)
            total += value
            count += 1
        assert count == 363
        return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=[1, 2, 4], default=4)
    args = parser.parse_args()
    d = Data()
    print("INITIALIZING_FULL_MODEL", args.batch, flush=True)
    engine = Engine(d, args.batch)
    assert not (
        OUT / "train/initialization.json"
    ).exists(), "Formal initialization already exists; explicit reconciliation required"
    write_json(
        OUT / "train/initialization.json",
        dict(
            initialization="random_from_scratch",
            pretrained_loaded=False,
            parameters=engine.parameter_count,
            seed=CFG["seed"],
            updates_per_epoch=456,
            total_updates=22800,
            batch=args.batch,
            accumulation=engine.accum,
            global_batch=32,
            devices=[d.device_kind for d in jax.local_devices()],
        ),
    )
    best = float("inf")
    update = 0
    for epoch in range(1, CFG["epochs"] + 1):
        start = time.time()
        ids = epoch_indices(d.centers("train"), epoch)
        losses = []
        grad = 0.0
        for begin in range(0, len(ids), 32):
            loss, grad = engine.update(ids[begin : begin + 32], update)
            losses.append(loss)
            update += 1
            if update == 1 or update % 50 == 0:
                print(
                    "TRAIN_STEP_OK",
                    json.dumps(
                        dict(
                            epoch=epoch,
                            update=update,
                            batch=begin // 32 + 1,
                            loss=loss,
                            grad_norm=grad,
                            used_lr=float(engine.schedule(update - 1)),
                            next_lr=float(engine.schedule(update)),
                        )
                    ),
                    flush=True,
                )
        val = engine.valid()
        record = dict(
            epoch=epoch,
            updates=update,
            train_denoising_loss=float(np.mean(losses)),
            valid_denoising_loss=val,
            grad_norm=grad,
            lr=float(engine.schedule(update)),
            seconds=time.time() - start,
        )
        params = unrep(engine.params)
        if val < best:
            best = val
            save(OUT / "train/best.pkl", dict(params=params, epoch=epoch, validation_loss=val))
        save(
            OUT / "train/last.pkl",
            dict(params=params, opt_state=unrep(engine.opt), epoch=epoch, updates=update),
        )
        record["best_valid"] = best
        with (OUT / "train/metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print("EPOCH_COMPLETE", json.dumps(record), flush=True)
    write_json(OUT / "train/TRAIN_COMPLETE", dict(epochs=50, updates=update))


if __name__ == "__main__":
    main()
