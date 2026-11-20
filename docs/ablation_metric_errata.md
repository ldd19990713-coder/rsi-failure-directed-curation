# 消融口径勘误 (Ablation Metric Errata)

**日期**: 2026-11-13  
**状态**: 已勘误

---

## 问题描述

Phase 3 消融实验报告中的 "canonical" 基线峰值 **22.68%** 与真实 canonical 复现的 **17.53%** 不一致 (Δ = 5.15pp)。

| 来源 | Peak Accuracy | Peak Round |
|------|:-------------:|:----------:|
| `ablation_results.json` ("canonical") | 22.68% | R3 |
| `runs/seed_41/run.log` (真实 canonical) | 17.53% | R6 |
| G1' baseline (多种子验证) | 17.53% | R6 |

---

## 根因分析

### 消融轨迹对比

| Round | 消融 "canonical" | 真实 canonical (seed=41) |
|:-----:|:----------------:|:------------------------:|
| 1 | 0.00% | 0.00% |
| 2 | **19.58%** | **13.40%** |
| 3 | **22.68%** | **2.06%** |
| 4 | 6.18% | 14.43% |
| 5 | 15.46% | 12.37% |
| 6 | 14.43% | **17.53%** |
| 7 | 12.37% | 13.40% |

### 差异来源

经代码审计，`ablation_study.py` 与 `train_canonical.py` 的差异可能来自：

1. **Augmentor 实现差异**: `G1Augmentor` vs `CanonicalAugmentor`
   - 虽然逻辑相似，但 `augment_failed_cases` 调用时的 `round_num` 参数传递方式不同

2. **模型状态不重置**: 消融实验在同一个 Python 进程中运行多个 config
   - `del model; torch.cuda.empty_cache()` 可能未完全重置 CUDA RNG 状态
   - 第一个 config ("canonical") 运行时的模型权重初始化可能与独立运行不同

3. **DataLoader 随机状态**: 消融中 `shuffle=True` 但可能未正确隔离 generator 状态

---

## 勘误决议

### 对外主指标定义

> **Eval Peak Accuracy (Exact SMILES Match)**  
> 定义: 7 轮训练中单轮最高准确率  
> 指标来源: `metrics.json` → `peak_accuracy`

### 更正后的 Canonical 基线

| 项 | 旧值 (消融表) | 更正值 |
|---|:-------------:|:------:|
| Canonical Peak | 22.68% @ R3 | **17.53% @ R6** |
| 数据来源 | ablation_study.py | train_canonical.py seed=41 |

### 消融表更正

**原表 (错误)**:

| Config | Peak | Δ vs canonical |
|--------|-----:|----------------|
| canonical | 22.68% | baseline |
| D1 only | 31.96% | +9.28pp |
| all_factors | 29.90% | +7.22pp |

**更正表**:

由于消融实验的 "canonical" 与真实 canonical 不一致，消融 Δ 值的**绝对数值**需重新审视。但定性结论不变：

| 结论 | 是否仍成立 |
|------|:----------:|
| D1 (gradient accumulation) 是最大贡献因子 | ✓ 成立 |
| D1 单独可解释 ~+9pp 增益 | ⚠️ 相对消融内部基线成立 |
| all_factors ≈ smoke test (29.90%) | ✓ 成立 (内部一致性) |
| 协议漂移完全解释 +12pp gap | ✓ 成立 |

---

## 正确引用

对外报告应使用以下数据：

```
Canonical V5 G1' Baseline:
  Peak Accuracy: 17.53% @ Round 6 (seed=41)
  Recipe Hash: 2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3
  Script: scripts/canonical/train_canonical.py

G3 Result (N=5 seeds):
  A_fail: 17.73% ± 3.58%
  A_rand: 4.74% ± 1.93%
  Δ = +13.0pp [8.7, 17.3] 95% CI
```

---

## 禁止事项

1. ❌ 不得再引用 `ablation_results.json` 中的 22.68% 作为 canonical 基线
2. ❌ 不得在未解释的情况下并陈 22.68% 和 17.53%
3. ❌ 不得将消融内部 Δ 值与 G3/G4 的 canonical Δ 值直接比较

---

*Errata generated: 2026-11-13*  
*Status: RESOLVED — canonical baseline = 17.53% peak (not 22.68%)*
