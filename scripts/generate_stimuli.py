#!/usr/bin/env python3
"""Generate the ItemFamily stimulus set (thin CLI over stimuli.generate).

Usage: generate_stimuli.py --config configs/default.yaml [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.stimuli.generate import generate_families, save_families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--out", type=Path, default=None, help="output JSONL (default: config paths.stimuli)"
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    families = generate_families(config)
    out = args.out if args.out is not None else Path(config.paths.stimuli)
    save_families(families, out)
    print(f"wrote {len(families)} families -> {out}")


if __name__ == "__main__":
    main()
