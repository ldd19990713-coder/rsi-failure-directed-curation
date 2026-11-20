#!/usr/bin/env python3
"""
Public entrypoint for LoRA fine-tuning.

Wraps the locked canonical / P2b runners with repo-relative paths.

Examples:
  python scripts/train_lora.py --config configs/recipe_7b.yaml --strategy a_fail --seed 41
  python scripts/train_lora.py --config configs/recipe_1.5b.yaml --strategy a_rand --seed 42
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML recipe (configs/recipe_7b.yaml | recipe_1.5b.yaml)")
    ap.add_argument("--strategy", choices=["a_fail", "a_rand"], default="a_fail")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--data", default="data/v9_g1prime")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = (ROOT / args.config).resolve()
    if not cfg.exists():
        print(f"Config not found: {cfg}", file=sys.stderr)
        return 2

    is_7b = "7b" in cfg.name.lower() or "7B" in cfg.read_text(encoding="utf-8")
    runner = ROOT / (
        "scripts/canonical/p2b_7b_experiment.py" if is_7b else "scripts/canonical/train_canonical.py"
    )
    out = args.output_dir or f"runs/seed{args.seed}_{args.strategy}"
    env = os.environ.copy()
    env["RSI_ROOT"] = str(ROOT)
    env["RSI_STRATEGY"] = args.strategy
    env["RSI_SEED"] = str(args.seed)
    env["RSI_DATA"] = str(ROOT / args.data)
    env["RSI_OUTPUT"] = str(ROOT / out)
    env["RSI_CONFIG"] = str(cfg)

    cmd = [
        sys.executable,
        str(runner),
        "--seed",
        str(args.seed),
        "--strategy",
        args.strategy,
    ]
    # Runners differ in CLI; prefer env-contract + best-effort flags.
    print("RSI_ROOT=", env["RSI_ROOT"])
    print("Runner:", runner.relative_to(ROOT))
    print("Note: legacy runners were written for an internal layout;")
    print("      paths are remapped via RSI_ROOT. If flags differ, call the")
    print("      canonical script directly after inspecting --help.")
    if args.dry_run:
        print("DRY-RUN cmd:", " ".join(cmd))
        return 0
    if not runner.exists():
        print(f"Missing runner: {runner}", file=sys.stderr)
        return 2
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    sys.exit(main())
