from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
old = "EXPERIMENT_DIR / \"configs\" / \"canonical\""
new = "EXPERIMENT_DIR / \"configs\""
old2 = "EXPERIMENT_DIR / 'configs'"
new2 = "EXPERIMENT_DIR / 'configs'"
n = 0
for p in ROOT.rglob("*.py"):
    t = p.read_text(encoding="utf-8")
    nt = t.replace(old, new).replace(old2, new2)
    if nt != t:
        p.write_text(nt, encoding="utf-8")
        n += 1
        print(p.relative_to(ROOT))
print("fixed", n)
