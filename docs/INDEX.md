# P0 Export 文件清单

**日期**: 2026-11-13  
**内容**: Phase 3 Canonical 实验完整产出

## 文件列表

| 文件 | 说明 |
|------|------|
| `P0_immediate_runbook.md` | P0 执行清单 (原始) |
| `Phase3_Canonical_Report.md` | Phase 3 综合报告 |
| `README.md` | 项目入口文档 |
| `ablation_metric_errata.md` | 消融口径勘误 (22.68% → 17.53%) |
| `phase3_g3g4_full_metrics.md` | G3/G4 全指标表 |
| `metrics_extract.csv` | 指标 CSV (20行, N=5 seeds × 4 arms) |
| `recipe_hash.txt` | Canonical 配方 hash |
| `v5_g1prime_canonical.yaml` | Canonical 配置文件 |
| `v9_baseline_lock.md` | V9 基线锁定 (含附录 A) |

## 关键结论

| 实验 | 结果 |
|------|------|
| G3: A_fail vs A_rand | **+13.0pp** (N=5, p<0.01) |
| G4: B_recur vs B_single | +1.65pp (边际) |

**Recipe Hash**: `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3`

## 状态

✅ P0 出口自检全部通过，可进入机制实验阶段
