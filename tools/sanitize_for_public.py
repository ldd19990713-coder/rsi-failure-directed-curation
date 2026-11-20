#!/usr/bin/env python3
"""One-shot sanitizer: strip internal absolute paths from the release tree."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTERNAL_ROOT = "/home/scientist/project2115974686745874434/ResearchSpace/RSI_MVE"
HF_1P5 = (
    "/home/scientist/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B/snapshots/"
)
HF_7B = (
    "/home/scientist/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B/snapshots/"
)

IMPORT_BLOCK = (
    "from pathlib import Path\n"
    "import os, sys\n"
    "_ROOT = Path(__file__).resolve()\n"
    "for _p in _ROOT.parents:\n"
    "    if (_p / 'LICENSE').exists() and (_p / 'CITATION.cff').exists():\n"
    "        sys.path.insert(0, str(_p))\n"
    "        break\n"
    "from paths import get_repo_root, model_id_or_path\n"
    "EXPERIMENT_DIR = get_repo_root()\n"
)


def depth_parents(path: Path) -> int:
    rel = path.relative_to(ROOT)
    return len(rel.parts) - 1


def sanitize_text(text: str, file_path: Path) -> str:
    text = text.replace("sys.path.insert(0, '/tmp/rsi_mve_libs')", "pass  # public package: no /tmp lib path")
    text = text.replace('sys.path.insert(0, "/tmp/rsi_mve_libs")', "pass  # public package: no /tmp lib path")

    # Absolute experiment root
    if INTERNAL_ROOT in text:
        text = text.replace(
            f'Path("{INTERNAL_ROOT}")',
            "get_repo_root()",
        )
        text = text.replace(
            f"Path('{INTERNAL_ROOT}')",
            "get_repo_root()",
        )
        text = text.replace(INTERNAL_ROOT, "get_repo_root()")  # leftover string uses

    # Model caches → HF ids (string literals in assignments)
    text = re.sub(
        r'LOCAL_MODEL_CACHE\s*=\s*"[^"]*Qwen2\.5-1\.5B[^"]*"',
        'LOCAL_MODEL_CACHE = model_id_or_path("Qwen/Qwen2.5-1.5B")',
        text,
    )
    text = re.sub(
        r'LOCAL_MODEL_CACHE\s*=\s*"[^"]*Qwen2\.5-7B[^"]*"',
        'LOCAL_MODEL_CACHE = model_id_or_path("Qwen/Qwen2.5-7B")',
        text,
    )
    text = re.sub(
        r'["\']' + re.escape(HF_1P5) + r'[^"\']*["\']',
        '"Qwen/Qwen2.5-1.5B"',
        text,
    )
    text = re.sub(
        r'["\']' + re.escape(HF_7B) + r'[^"\']*["\']',
        '"Qwen/Qwen2.5-7B"',
        text,
    )
    # Any remaining /home/scientist paths
    text = re.sub(r'["\']/home/scientist[^"\']*["\']', '"."', text)

    if "get_repo_root" in text or "model_id_or_path" in text:
        if "from paths import" not in text:
            # Inject import after shebang/docstring if present
            lines = text.splitlines(keepends=True)
            insert_at = 0
            if lines and lines[0].startswith("#!"):
                insert_at = 1
            if insert_at < len(lines) and lines[insert_at].startswith('"""'):
                # skip docstring
                insert_at += 1
                while insert_at < len(lines) and '"""' not in lines[insert_at - 1][3:] and not (
                    lines[insert_at - 1].strip().endswith('"""') and lines[insert_at - 1].count('"""') >= 2
                ):
                    if '"""' in lines[insert_at - 1] and insert_at > 1:
                        break
                    insert_at += 1
                    if insert_at > 1 and '"""' in lines[insert_at - 1]:
                        break
            # Simpler: prepend after first blank line following module docstring
            m = re.match(r'(#!.*\n)?("""[\s\S]*?"""\n|\'\'\'[\s\S]*?\'\'\'\n)?', text)
            end = m.end() if m else 0
            inject = (
                "\nimport sys\n"
                "from pathlib import Path\n"
                "_here = Path(__file__).resolve()\n"
                "for _p in _here.parents:\n"
                "    if (_p / 'LICENSE').exists() and (_p / 'CITATION.cff').exists():\n"
                "        if str(_p) not in sys.path:\n"
                "            sys.path.insert(0, str(_p))\n"
                "        break\n"
                "from paths import get_repo_root, model_id_or_path  # noqa: E402\n\n"
            )
            text = text[:end] + inject + text[end:]
    return text


def sanitize_yaml(text: str) -> str:
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
    text = re.sub(r'/home/scientist[^\s"]*', "Qwen/Qwen2.5-1.5B", text)
    return text


def main() -> None:
    n = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".md", ".txt"}:
            continue
        if "tools/sanitize" in str(path).replace("\\", "/"):
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix in {".yaml", ".yml"}:
            new = sanitize_yaml(raw)
        elif path.suffix == ".py":
            new = sanitize_text(raw, path)
        else:
            new = raw.replace(INTERNAL_ROOT, "<REPO_ROOT>")
            new = re.sub(r"/home/scientist/[^\s)`'\"]+", "<INTERNAL_PATH_REDACTED>", new)
        if new != raw:
            path.write_text(new, encoding="utf-8")
            n += 1
            print(f"sanitized: {path.relative_to(ROOT)}")
    print(f"done: {n} files")


if __name__ == "__main__":
    main()
