"""Terminal fair-CRPS training; validation selection never reads test data."""

import argparse, os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/crps/one_day.yaml")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    os.environ["WEATHERCRAFT_CRPS_CONFIG"] = a.config
    from .training import main as train

    train(a.resume)


if __name__ == "__main__":
    main()
