#!/usr/bin/env python3
"""Generate a JSON embedding vector for config testing."""

from __future__ import annotations

import argparse
import json
import random


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["zeros", "random"], default="zeros")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.mode == "zeros":
        vector = [0.0] * args.dim
    else:
        vector = [random.random() for _ in range(args.dim)]

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(vector, handle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
