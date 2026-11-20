#!/usr/bin/env python3
"""
Verify that the clean holdout set has zero reaction-key overlap with train∪eval.

Usage:
    python evaluation/key_disjoint_check.py \\
        --train data/v9_g1prime/train.json \\
        --eval data/v9_g1prime/eval.json \\
        --holdout data/holdout_clean/holdout_n31.json

Expected: 0 overlapping reaction keys.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Set


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    return data


def reaction_key(rec: Dict[str, Any]) -> str:
    if rec.get("reaction_key"):
        return str(rec["reaction_key"]).strip()
    reactants = rec.get("reactants_smiles") or []
    if isinstance(reactants, str):
        left = reactants
    else:
        left = ".".join(str(x) for x in reactants)
    product = (rec.get("product_smiles") or rec.get("target_text") or "").strip()
    return f"{left}>>{product}"


def extract_keys(records: List[Dict[str, Any]]) -> Set[str]:
    return {reaction_key(r) for r in records if reaction_key(r)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--holdout", required=True)
    args = ap.parse_args()

    train = load_json(args.train)
    eval_ = load_json(args.eval)
    holdout = load_json(args.holdout)

    te_keys = extract_keys(train) | extract_keys(eval_)
    ho_keys = extract_keys(holdout)
    overlap = te_keys & ho_keys

    print(f"Train keys: {len(extract_keys(train))}")
    print(f"Eval keys:  {len(extract_keys(eval_))}")
    print(f"Train∪Eval: {len(te_keys)}")
    print(f"Holdout:    {len(ho_keys)}")
    print(f"Overlap:    {len(overlap)}")
    if overlap:
        print("FAIL: reaction-key overlap detected:")
        for k in sorted(overlap):
            print(f"  {k}")
        return 1
    print("PASS: zero reaction-key overlap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
