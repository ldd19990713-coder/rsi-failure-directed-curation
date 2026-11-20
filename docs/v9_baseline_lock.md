# V9 作战基线锁定（G1'-R1）

**锁定日期**: 2026-11-08（以正确报告为准）  
**源报告**: `docs/G1PRIME_FINAL_N10_REPORT.md`

## 协议身份

| 项 | 值 |
| :--- | :--- |
| Split | `v9_g1prime` |
| Train / Eval | 46 / **97**（目标≥100，差 3 **书面豁免**） |
| 配置 | V5 frozen（wd=0.01, eta_min_ratio=0.1, 7 rounds） |
| 种子 | 41–50（N=10） |
| 评估 | 精确 SMILES；Invalid=0% |

## 锁定数字

| 指标 | 值 |
| :--- | :--- |
| 峰值准确率 | **14.43% ± 2.21%**（SE **0.70%**） |
| 95% CI | **[13.06%, 15.80%]** |
| 平均回撤 | 7.22% |
| G2 | **PASS** |
| σ' 目标 (&lt;5–6%) | **达标（2.21%）** |

## 双层状态

- **L1 绝对复现窗 [33.3%, 46.7%]**:** FAIL**（作废 40%/29% 通过叙事）  
- **L2 作战基线**:** LOCKED** → 允许相对 P1  

## 允许 / 禁止

- ✅ P1：G3/G4 相对本基线  
- ❌ 调窗冒充 L1 通过；❌ 改 weight_decay；❌ 未归档主张「RSI 已验证」

---

## 附录 A — Canonical 配方与 Phase3（2026-11-13）

### 唯一合法入口

| 项 | 值 |
| :--- | :--- |
| 训练脚本 | `scripts/canonical/train_canonical.py` |
| Config | `configs/canonical/v5_g1prime_canonical.yaml` |
| Recipe Hash | `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3` |
| 废弃脚本 | `scripts/_DEPRECATED_smoke_g1prime_train.py.bak` (D1/D2/D3 漂移) |

### Phase 3 核心结果

| 实验 | 对比 | Δ | 结论 |
| :--- | :--- | :--- | :--- |
| G3 | A_fail (17.73%) vs A_rand (4.74%) | **+13.0pp** | **支持** A_fail >> A_rand |
| G4 | B_recur (17.73%) vs B_single (16.08%) | +1.65pp | **边际** 不稳定 |

### 主张模板

> 在 v9 无泄露 + V5 canonical 协议下，失败驱动增广相对等量随机增广显著提升峰值准确率（约 +13pp，N=5）；等预算下递归多轮相对单训仅有小且不稳定增益（约 +1.7pp）。结果支持数据中心有界自精炼，不支持将「递归调度」表述为充分的持续自我改进机制。

### 禁止句

- ❌ 不得使用 smoke 29.9% 作为基线
- ❌ 不得使用旧 P1 Δ≈0 结论
- ❌ 不得写「持续 RSI 已验证」
- ❌ 不得引用消融表 22.68%（见 `docs/ablation_metric_errata.md`）

### P0 出口自检

- [x] 唯一入口 = canonical；漂移脚本已 DEPRECATED  
- [x] Recipe hash 与报告一致  
- [x] 消融 22.68 vs 17.53 已书面勘误 (`docs/ablation_metric_errata.md`)  
- [x] G3/G4 全指标表已归档 (`docs/phase3_g3g4_full_metrics.md`)  
- [x] 主张模板已写入 baseline lock 附录  
- [x] 无新的 GPU 实验混入未冻结因子
