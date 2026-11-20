#!/usr/bin/env python3
"""
Three-level leakage audit for reaction splits.

Checks:
  1. ID overlap
  2. Product SMILES overlap (informational; not always leakage)
  3. Full reaction-key overlap (must be 0 between disjoint splits)

Usage:
  python evaluation/leakage_audit.py --data data/v9_g1prime
  python evaluation/leakage_audit.py \\
      --train data/v9_g1prime/train.json \\
      --eval data/v9_g1prime/eval.json \\
      --holdout data/holdout_clean/holdout_n31.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
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


def product_smiles(rec: Dict[str, Any]) -> str:
    return (rec.get("product_smiles") or rec.get("target_text") or "").strip()


def ids(recs: Iterable[Dict[str, Any]]) -> Set[str]:
    return {str(r.get("id")) for r in recs if r.get("id") is not None}


def report_pair(name_a: str, a: List[Dict], name_b: str, b: List[Dict]) -> Tuple[bool, Dict[str, Any]]:
    id_overlap = ids(a) & ids(b)
    prod_a = {product_smiles(r) for r in a if product_smiles(r)}
    prod_b = {product_smiles(r) for r in b if product_smiles(r)}
    prod_overlap = prod_a & prod_b
    key_a = {reaction_key(r) for r in a}
    key_b = {reaction_key(r) for r in b}
    key_overlap = key_a & key_b
    ok = len(id_overlap) == 0 and len(key_overlap) == 0
    return ok, {
        "pair": f"{name_a} vs {name_b}",
        "id_overlap": len(id_overlap),
        "product_smiles_overlap": len(prod_overlap),
        "reaction_key_overlap": len(key_overlap),
        "pass": ok,
        "id_overlap_samples": sorted(id_overlap)[:20],
        "reaction_key_overlap_samples": sorted(key_overlap)[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, help="Directory with train.json + eval.json")
    ap.add_argument("--train", type=Path)
    ap.add_argument("--eval", type=Path)
    ap.add_argument("--holdout", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.data:
        train_path = args.data / "train.json"
        eval_path = args.data / "eval.json"
    else:
        train_path, eval_path = args.train, args.eval
    if not train_path or not eval_path:
        ap.error("Provide --data DIR or both --train and --eval")

    train = load_json(train_path)
    eval_ = load_json(eval_path)
    results = []
    all_ok = True

    ok, r = report_pair("train", train, "eval", eval_)
    results.append(r)
    all_ok &= ok
    print(json.dumps(r, ensure_ascii=False, indent=2))

    if args.holdout:
        holdout = load_json(args.holdout)
        for name, split in (("train", train), ("eval", eval_), ("train∪eval", train + eval_)):
            ok, r = report_pair(name, split if name != "train∪eval" else train + eval_, "holdout", holdout)
            results.append(r)
            all_ok &= ok
            print(json.dumps(r, ensure_ascii=False, indent=2))

    summary = {"pass": all_ok, "checks": results}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
