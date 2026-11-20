#!/usr/bin/env python3
"""Ensure train/eval JSON records include a reaction_key field (in-place)."""
from __future__ import annotations

import json
from pathlib import Path


def key_for(rec: dict) -> str:
    if rec.get("reaction_key"):
        return str(rec["reaction_key"]).strip()
    reactants = rec.get("reactants_smiles") or []
    if isinstance(reactants, str):
        left = reactants
    else:
        left = ".".join(str(x) for x in reactants)
    product = (rec.get("product_smiles") or rec.get("target_text") or "").strip()
    return f"{left}>>{product}"


def patch(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for rec in data:
        k = key_for(rec)
        if rec.get("reaction_key") != k:
            rec["reaction_key"] = k
            n += 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return n


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "data/v9_g1prime/train.json",
        "data/v9_g1prime/eval.json",
        "data/holdout_clean/holdout_n31.json",
    ):
        path = root / rel
        n = patch(path)
        print(f"{rel}: updated {n} reaction_key fields")


if __name__ == "__main__":
    main()
