"""Create deterministic online model weights for code integration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.online_predictor import OnlineLogisticPredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--scale", type=float, default=0.01)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("model already exists")
    model = OnlineLogisticPredictor.seeded_random(
        seed=args.seed, dimension=256, scale=args.scale
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"wrote fixed online model: {args.output.resolve()}")


if __name__ == "__main__":
    main()
