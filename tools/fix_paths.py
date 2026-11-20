#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = "/home/scientist/project2115974686745874434/ResearchSpace/RSI_MVE"


def transform(text: str) -> str:
    text = text.replace("sys.path.insert(0, '/tmp/rsi_mve_libs')", "pass  # public: no /tmp libs")
    text = text.replace('sys.path.insert(0, "/tmp/rsi_mve_libs")', "pass  # public: no /tmp libs")
    walk = (
        "next(p for p in Path(__file__).resolve().parents "
        "if (p / 'CITATION.cff').exists())"
    )
    text = text.replace(f'Path("{INTERNAL}")', walk)
    text = text.replace(f"Path('{INTERNAL}')", walk)
    text = text.replace(INTERNAL, ".")
    text = re.sub(
        r'LOCAL_MODEL_CACHE\s*=\s*"[^"]*Qwen2\.5-1\.5B[^"]*"',
        'LOCAL_MODEL_CACHE = "Qwen/Qwen2.5-1.5B"',
        text,
    )
    text = re.sub(
        r'LOCAL_MODEL_CACHE\s*=\s*"[^"]*Qwen2\.5-7B[^"]*"',
        'LOCAL_MODEL_CACHE = "Qwen/Qwen2.5-7B"',
        text,
    )
    text = re.sub(
        r'path:\s*"[^"]*Qwen2\.5-1\.5B[^"]*"',
        'path: "Qwen/Qwen2.5-1.5B"',
        text,
    )
    text = re.sub(
        r'path:\s*"[^"]*Qwen2\.5-7B[^"]*"',
        'path: "Qwen/Qwen2.5-7B"',
        text,
    )
    text = text.replace(
        'train_path: "data/v9_g1prime/train_reactions_g1prime.json"',
        'train_path: "data/v9_g1prime/train.json"',
    )
    text = text.replace(
        'eval_path: "data/v9_g1prime/eval_reactions_g1prime.json"',
        'eval_path: "data/v9_g1prime/eval.json"',
    )
    text = re.sub(r'"/home/scientist[^"]*"', '"."', text)
    text = re.sub(r"'/home/scientist[^']*'", "'.'", text)
    return text


def main() -> None:
    n = 0
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in {".py", ".yaml", ".yml", ".md", ".txt"}:
            continue
        if p.name in {"fix_paths.py", "sanitize_for_public.py"}:
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        new = transform(raw)
        if new != raw:
            p.write_text(new, encoding="utf-8")
            n += 1
            print(p.relative_to(ROOT))
    print(f"updated {n} files")
    leftover = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "/home/scientist" in t:
            leftover.append(str(p.relative_to(ROOT)))
    print("leftover /home/scientist:", len(leftover))
    for x in leftover[:20]:
        print(" ", x)


if __name__ == "__main__":
    main()
