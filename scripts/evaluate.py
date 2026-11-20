#!/usr/bin/env python3
"""
Exact-SMILES evaluation helper (public entrypoint).

Computes accuracy of predicted product SMILES against a JSON split.
Supports evaluating a JSONL of predictions or a simple adapter smoke path.

Usage:
  python scripts/evaluate.py --split data/v9_g1prime/eval.json --preds runs/preds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "predictions" in data:
            return data["predictions"]
        if "samples" in data:
            return data["samples"]
    return data


def canon(s: str) -> str:
    return (s or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="Gold JSON split")
    ap.add_argument("--preds", required=True, help="Predictions JSON (id -> pred or list with id/pred)")
    args = ap.parse_args()

    gold = {str(r["id"]): canon(r.get("product_smiles") or r.get("target_text")) for r in load_records(Path(args.split))}
    raw = load_records(Path(args.preds))
    if isinstance(raw, dict) and all(not isinstance(v, dict) for v in raw.values()):
        preds = {str(k): canon(v) for k, v in raw.items()}
    else:
        preds = {}
        for r in raw:
            preds[str(r.get("id"))] = canon(r.get("prediction") or r.get("pred") or r.get("product_smiles"))

    n = 0
    correct = 0
    missing = 0
    for gid, g in gold.items():
        n += 1
        if gid not in preds:
            missing += 1
            continue
        if preds[gid] == g:
            correct += 1

    acc = 100.0 * correct / n if n else 0.0
    print(json.dumps({
        "n": n,
        "correct": correct,
        "missing": missing,
        "exact_smiles_accuracy_pct": round(acc, 4),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
