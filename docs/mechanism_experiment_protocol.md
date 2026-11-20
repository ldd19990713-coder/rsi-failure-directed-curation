# 机制实验协议（P0 出口后）

**生效前提**: `P0_export_20261113` 自检通过（见裁决下文）。  
**唯一入口**: `scripts/canonical/train_canonical.py` + `v5_g1prime_canonical.yaml`  
**Hash**: `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3`  
**主指标**: eval **peak** exact-SMILES；辅指标: final、drawdown、E/M/H；invalid 若脚本未记则标 N/A 并计划补记。

---

## 0. 总判决：可以进入机制实验

| P0 出口项 | 证据（本包） | 状态 |
| :--- | :--- | :---: |
| 唯一入口 / 漂移 DEPRECATED | `README.md` Deprecated 节 | ✅ |
| recipe hash 一致 | `recipe_hash.txt` = 报告 hash | ✅ |
| 消融 22.68 vs 17.53 勘误 | `ablation_metric_errata.md` | ✅ |
| G3/G4 全指标归档 | `phase3_g3g4_full_metrics.md` + CSV（20 行） | ✅ |
| 主张写入 baseline lock | `v9_baseline_lock.md` 附录 A | ✅ |
| 无未冻因子混跑 | 出口声明 | ✅ |

**非阻塞缺口**（机制阶段并行补，不挡开跑）:

- `invalid_rate` = N/A（脚本未追踪）→ 机制跑次起尽量补记  
- 消融绝对 Δ 不可与 G3 直接比 → **机制实验禁止依赖消融绝对值**  
- 消融进程内多 config 污染已记录 → 机制实验 **一进程一 config**

---

## 1. 硬约束（违反即作废该跑次）

1. **不改** precision / LR 公式 `0.85^(r-2)` / batch=2 / wd / eta_min / rounds=7 / 英增广 5 类集合（除非该实验因子本身就是「去掉某一类」）。  
2. **禁止** `_DEPRECATED_smoke_*.bak`、grad-accum、`0.85^(r-1)`、`round+1`。  
3. **一次只动一个因子**；不得并行扩 train / 换 7B。  
4. 每跑次启动时打印并归档 `recipe_sha256`；结束后写 `metrics.json`（至少 peak/final/drawdown/E/M/H）。  
5. 种子默认 **41,42,43**（筛选用）；阳性后再扩 **41–45**；投稿稳健性再 **46–50**。  
6. 对照臂必须 **等量**（增广条数 / 训练步数 / 轮次预算按假说定义匹配）。  
7. 对外仍禁止：「持续 RSI 已验证」；禁止引用 22.68% / smoke 29.9% / 旧 P1 Δ≈0。

---

## 2. 实验包排序（建议 2–3 周）

### M1 — 失败变体 leave-one-out（优先，回答「哪类变体在起作用」）

**假说**: 去掉某一 prompt 变体类型后，A_fail 峰值显著下降 → 该类必要。

| 项 | 规定 |
| :--- | :--- |
| 基座 | canonical + **A_fail**（失败驱动触发不变） |
| 因子 | 5 类中每次去掉 1 类：condition / hint / format / synonym / context |
| 对照 | **Full-5**（完整 A_fail，可复用 Phase3 G3 A_fail 同种子，不必重跑若 hash 一致） |
| 种子 | 先 41–43；若某类 Δ&lt;2pp 再扩 41–45 |
| 等量 | 去掉一类后，该难度仍按 yaml 的 `num_variants` **从剩余类型取前 k 个**（写进 manifest）；禁止偷偷加噪声 token |
| 产出 | 每类：mean peak、vs Full-5 的 Δ、E/M/H；一张 leave-one-out 表 |
| 通过解读 | 必要类：Δ≤−3pp 且 ≥2/3 种子同向；否则标「可替换/弱」 |

**命令骨架**（远端实现后）:

```bash
# 示例：禁用 synonym_variant
python scripts/canonical/train_canonical.py \
  --config configs/canonical/v5_g1prime_canonical.yaml \
  --seed 41 \
  --aug-drop synonym_variant \
  --tag M1_loo_synonym
# 每个 (drop_type × seed) 独立进程；禁止与 ablation_study 多 config 同进程
```

### M2 — 等量增广曲线（回答「是失败定向，还是只是更多相关数据」）

**假说**: 在 **相同增广条数** 下，A_fail 仍优于 A_rand。

| 项 | 规定 |
| :--- | :--- |
| 臂 | A_fail vs A_rand |
| 因子 | 每轮增广预算 K ∈ {0.5×, 1×, 2×} 相对现行 G3 默认量（先定义「1×=现行」并写进 config tag） |
| 匹配 | 同 seed、同 K；A_rand 从 train 池随机抽 **恰好 K 条**（非失败定向） |
| 种子 | 41–43（曲线），确认后再 41–45 |
| 产出 | 横轴=K，纵轴=peak；两条曲线 + Δ(K) |
| 解读 | 若全 K 上 A_fail>A_rand → 定向效应稳健；若仅高 K 拉开 → 量效应为主 |

### M3 — 采纳/回滚（回答「能否在保住 A_fail 增益下压 drawdown」）

**动机**: Phase3 A_fail mean drawdown **≈12.6%**；B_recur final **低于** B_single final（≈−5pp）→ 多轮有过拟合/回撤问题。

| 项 | 规定 |
| :--- | :--- |
| 数据 | **固定** 与 G3 A_fail 相同的失败驱动策略（或冻结某 seed 的 aug manifest 重放） |
| 因子 | 每轮结束后：若 eval peak（或 rolling best）下降 ≥ τ → **回滚权重**到上轮；τ ∈ {0, 1/97} 先取 τ=0（变差即拒） |
| 对照 | 无回滚的 B_recur/A_fail（Phase3 可复用） |
| 种子 | 41–45 |
| 主看 | drawdown ↓、final ↑；peak 允许小幅牺牲（预先声明：peak 掉 &gt;2pp 则策略失败） |
| 禁止 | 同时改 LR 或变体集合 |

### M4 —（可选，不与 M1–M3 并行）失败覆盖率

统计每轮失败样本被增广覆盖的比例、重复失败是否反复增广；与 peak 做相关。偏分析、少重训。

---

## 3. 单次跑次检查单（复制到笔记）

```
[ ] recipe_sha256 打印 == 2a965c39…391dc3
[ ] 仅改本实验声明的一个因子
[ ] 独立进程；CUDA RNG / 模型从基座 reload
[ ] aug_manifest.json 含：trigger、types、dropped、K、seed、round
[ ] metrics: peak, peak_round, final, drawdown, E/M/H (, invalid)
[ ] 未引用 smoke / 22.68 / 旧 P1
[ ] tag = M{1|2|3}_... 写入 artifacts/canonical/mechanism/
```

---

## 4. 阶段出口 → 才允许扩数据/7B

机制包最小完成定义：

1. **M1** 完成 Full-5 vs 至少 5 个 leave-one-out（seeds 41–43）并出表  
2. **M2** 至少 1× 与 2× 两点曲线（41–43）确认定向仍在  
3. **M3** 有无回滚对照的 drawdown/final 对比（41–45 或 41–43）  

之后才开：**train≥200** 或 7B+LoRA（仍锁失败驱动 + canonical 训练超参哲学）。

---

## 5. 明确不要做

| 禁止 | 原因 |
| :--- | :--- |
| 用 `ablation_study.py` 多 config 同进程做机制 | 已证实轨迹污染 |
| 先扩 7B 再做 M1 | 混因 |
| 把 M1–M3 因子叠在一次网格里 | 无法归因 |
| 用消融 D1 的 31.96% 当机制基线 | 非真实 canonical |

---

*基于 P0_export_20261113；本地裁决日期 2026-11-13*
