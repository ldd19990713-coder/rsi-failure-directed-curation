# P0 立刻可执行清单（Phase 3 后）

**目标**: 冻结唯一合法配方 → 消融口径一致 → G3/G4 全指标入档 → 主张模板落地。  
**不做**: 扩模、扩 train、重跑漂移配方、宣称「RSI 已验证」。  
**机器**: ResearchSpace / `RSI_MVE`（训练代码与 `artifacts/canonical/` 所在处）。  
**本机同步**: `D:\lidong\RSI-research\docs\`、`notes\`。

**配方**: `v5_g1prime_canonical`  
**Hash**: `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3`

---

## Day 0（今天，约 1–2h，无 GPU）

### Step 0.1 — 定位并核对真源

在 ResearchSpace 项目根执行：

```bash
cd /path/to/RSI_MVE   # 以实际路径为准

test -f scripts/canonical/train_canonical.py
test -f configs/canonical/v5_g1prime_canonical.yaml
cat configs/canonical/recipe_hash.txt
# 期望输出完整 hash 与上表一致

# 确认入口会打印 / 校验 hash
grep -n "recipe_hash\|2a965c39\|assert" scripts/canonical/train_canonical.py | head -40
```

**通过标准**: hash 文件存在且与报告一致；`train_canonical.py` 启动时校验或写入该 hash。

### Step 0.2 — 废弃漂移入口（硬隔离）

```bash
# 1) 找出所有漂移训练入口
rg -n "smoke_g1prime_train|grad.?accum|0\.85\^\(r-1\)|round\s*\+\s*1" \
  scripts/ configs/ --glob '!artifacts/**' || true

# 2) 对已知漂移脚本改名 + 顶栏警告（示例）
#    若文件在 scripts/ 下：
mv scripts/smoke_g1prime_train.py scripts/_DEPRECATED_smoke_g1prime_train.py.bak 2>/dev/null || true
# 在 .bak 文件首行插入：
# DEPRECATED — protocol drift (D1/D2/D3). DO NOT RUN. Use scripts/canonical/train_canonical.py

# 3) README / 启动说明只留一条合法命令
```

在 `README`（或项目入口文档）写入**唯一**命令模板：

```bash
python scripts/canonical/train_canonical.py \
  --config configs/canonical/v5_g1prime_canonical.yaml \
  --seed <SEED>
```

G3/G4：

```bash
python scripts/canonical/g3_g4_canonical.py --gate G3 --seed <SEED>
python scripts/canonical/g3_g4_canonical.py --gate G4 --seed <SEED>
# 具体 CLI 以脚本 --help 为准；禁止改精度/LR/batch/aug offset
```

**通过标准**: `rg smoke_g1prime_train` 在可执行路径无命中（仅 `.bak` / DEPRECATED）；新人按 README 只会进 canonical。

### Step 0.3 — 本机文档同步（可并行）

在 `D:\lidong\RSI-research`：

1. 打开 `docs/phase3_verdict.md` §3 主张模板，确认可直接粘贴。  
2. 在 `docs/v9_baseline_lock.md` **追加附录**（勿改已锁 G1' 数字），内容见 Step 2.2。  
3. 更新 `README.md`「当前核心交付物」指向 Phase3 / canonical（可选，与 0.2 一起做）。

---

## Day 1（今天–明天，约 2–4h，无 GPU 优先）

### Step 1.1 — 澄清消融「canonical = 22.68%」

**问题**: §1 复现 peak=**17.53%**；消融表 canonical=**22.68%**（等于 smoke 的 final）。

在 ResearchSpace：

```bash
# 读消融原始结果
python - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/canonical/ablation/ablation_results.json")
data = json.loads(p.read_text())
# 打印每个 config 的 peak / final / round 轨迹（字段名以实际 JSON 为准）
print(json.dumps(data, indent=2)[:8000])
PY

# 对照 seed41 复现 log
rg -n "acc|peak|final|Round" artifacts/canonical/runs/seed_41/run.log | head -80
```

**必做判定（二选一写进勘误）**:

| 情况 | 动作 |
| :--- | :--- |
| A. 消融脚本误用了 final / 错误 baseline 行 | 修表：canonical 行改为 **17.53% peak**；重算 Δ；发 `docs/ablation_errata.md` |
| B. 消融「canonical」实际不是同一入口 | 标红作废该行；用 §1 seed41 轨迹作唯一 canonical 对照；必要时 **只重跑消融的 canonical 对照 + D1**（seed41） |

**产出文件**（本机或远端均可）: `docs/ablation_metric_errata.md`，至少含：

- 指标定义：对外主指标 = **eval peak accuracy（精确 SMILES）**  
- 旧表哪一行错 / 为何 22.68 出现  
- 更正后的 canonical 行与 all_factors=29.90% 是否仍成立  
- 「D1 主导」结论是否仍成立（预期：方向不变）

**通过标准**: 任何对外表不再并陈未解释的 22.68% 与 17.53%。

### Step 1.2 — 从 metrics.json 抽全指标表（无重训）

```bash
# 列出 G3/G4 产物
find artifacts/canonical/g3_g4 -name 'metrics.json' | sort

python - <<'PY'
import json, csv
from pathlib import Path
root = Path("artifacts/canonical/g3_g4")
rows = []
for p in sorted(root.rglob("metrics.json")):
    m = json.loads(p.read_text())
    # 按实际字段适配；常见候选名如下
    rows.append({
        "path": str(p),
        "arm": p.parts[-3] if len(p.parts)>=3 else "",
        "seed": p.parts[-2],
        "peak": m.get("peak_acc", m.get("peak_accuracy")),
        "peak_round": m.get("peak_round"),
        "final": m.get("final_acc", m.get("final_accuracy")),
        "drawdown": m.get("drawdown", m.get("max_drawdown")),
        "invalid": m.get("invalid_rate", m.get("invalid")),
        "easy": m.get("acc_easy", m.get("E")),
        "medium": m.get("acc_medium", m.get("M")),
        "hard": m.get("acc_hard", m.get("H")),
        "raw_keys": ",".join(sorted(m.keys())[:40]),
    })
out = Path("artifacts/canonical/g3_g4/metrics_extract.csv")
out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
print("wrote", out, "n=", len(rows))
print("sample keys:", rows[0]["raw_keys"] if rows else None)
PY
```

**若缺字段**（drawdown / invalid / E/M/H）:

```bash
# 优先：从同目录 eval 明细或 run.log 重算（不重训）
# 次选：写一小段 offline recompute，读 predictions + gold
# 仅当无法离线重算时，才计划「只 eval、不 train」回放
```

**产出**:  
- `artifacts/canonical/g3_g4/metrics_extract.csv`  
- `docs/phase3_g3g4_full_metrics.md`（A_fail/A_rand/B_recur/B_single × seed41–45：peak、peak_round、final、drawdown、invalid、E/M/H）

**通过标准**: G3/G4 每个 seed×臂至少有 peak；缺项显式标 `N/A + 原因`，不假装有数。

---

## Day 2（文档冻结，约 1–2h）

### Step 2.1 — 主张模板落地

在以下文件**同一段话**（已写在 `docs/phase3_verdict.md` §3）:

- `docs/v9_baseline_lock.md` 附录  
- `notes/STATUS.md`（已有摘要则核对）  
- `docs/NEXT_STEPS.md`

主张（复制即用）:

> 在 v9 无泄露 + V5 canonical 协议下，失败驱动增广相对等量随机增广显著提升峰值准确率（约 +13pp，N=5）；等预算下递归多轮相对单训仅有小且不稳定增益（约 +1.7pp）。结果支持数据中心有界自精炼，不支持将「递归调度」表述为充分的持续自我改进机制。

**禁止句**写进 checklist 页脚：不得使用 smoke 29.9%、旧 P1 Δ≈0、「持续 RSI 已验证」。

### Step 2.2 — 更新 `v9_baseline_lock.md` 附录（模板）

追加（勿改正文锁定表）:

```markdown
## 附录 A — Canonical 配方与 Phase3（2026-11-13）

| 项 | 值 |
| :--- | :--- |
| 唯一训练入口 | `scripts/canonical/train_canonical.py` |
| Config | `configs/canonical/v5_g1prime_canonical.yaml` |
| Recipe hash | `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3` |
| 复现 | seed41–43 与 G1' 峰值精确匹配 |
| H3 (G3) | **支持** A_fail−A_rand ≈ +13.0pp（N=5） |
| H4 (G4) | **边际** B_recur−B_single ≈ +1.65pp（N=5） |
| 旧 P1 | **作废**（协议漂移伪阴性） |
| 废弃脚本 | `smoke_g1prime_train.py`（D1/D2/D3） |
```

### Step 2.3 — P0 出口自检（全部勾上才进入机制实验）

- [ ] 唯一入口 = canonical；漂移脚本已 DEPRECATED  
- [ ] recipe hash 与报告一致  
- [ ] 消融 22.68 vs 17.53 已书面勘误  
- [ ] G3/G4 全指标表已归档（缺项有说明）  
- [ ] 主张模板已写入 baseline lock 附录  
- [ ] 无新的 GPU 实验混入未冻结因子  

---

## 可选（不阻塞 P0 出口）

### Step O1 — G3 扩到 N=10（投稿稳健性）

```bash
# 仅 canonical；种子 46–50；只跑 G3 两臂
for s in 46 47 48 49 50; do
  python scripts/canonical/g3_g4_canonical.py --gate G3 --seed $s
done
# 合并后更新 mean±SD、符号一致性、CI
```

预算：约 5 seeds × 2 arms × 7 rounds（按历史 ~A100 小时估算后写入笔记）。  
**勿**同时改数据或 LR。

---

## 明确不要做（执行时对照）

| 禁止 | 原因 |
| :--- | :--- |
| 再跑 `smoke_g1prime_train.py` | D1/D2/D3 漂移 |
| 把 29.9% 写入基线表 | 非 canonical |
| 未勘误消融就开变体消融 | 口径污染 |
| 同时改失败类型 + 扩数据 + 7B | 混因 |
| 对外写「RSI / 持续自改进已验证」 | 证据不足 |

---

## 完成后立刻进入的下一包（P1，另开清单）

仅当上方 P0 出口全勾：

1. 失败变体类型 leave-one-out（canonical，固定 seed 子集）  
2. 等量增广曲线 A_fail vs A_rand  
3. 固定 A_fail 池上试采纳/回滚（降 drawdown）
