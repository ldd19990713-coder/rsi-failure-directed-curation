# 7B-LoRA 冻结配方 (V5 规范)

**生效日期**: 2026-11-18  
**实验代号**: P2b  
**数据**: v9_g1prime (train=46, eval=97, FROZEN)  
**7B Recipe Hash**: `9e3788dd7a2eb89d66a1b403d5c5b7c425d583adf7c799c35b761b57fbf7f809`

> ⚠️ **区分说明**: 上述 hash 为 7B-LoRA 专属（bf16/LoRA/bs=1），与 L2 (1.5B) 的 `2a965c...` 不同。二者不可混称。

---

## 1. 模型配置

| 参数 | 值 | 说明 |
|:---|:---|:---|
| 基座模型 | Qwen2.5-7B | 本地缓存路径见下 |
| 精度 | **bfloat16** | ⚠️ **禁用 fp16**（NaN 梯度问题） |
| Attention 实现 | `eager` | 避免 cuDNN frontend 问题 |
| 梯度检查点 | **禁用** | LoRA + gradient checkpointing 存在 cuDNN 兼容性问题 |

### LoRA 参数

| 参数 | 值 |
|:---|:---|
| `r` (rank) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |
| 可训练参数 | 40,370,176 / 7,655,986,688 (0.527%) |

### 本地模型路径

```
Qwen/Qwen2.5-7B  (Hugging Face hub id; or local path under ./models/Qwen2.5-7B)
```

---

## 2. 训练配置

| 参数 | 值 |
|:---|:---|
| batch_size | 1 (7B 显存限制) |
| max_length | 256 |
| epochs_per_round | 3 |
| num_rounds | 7 |
| 优化器 | AdamW |
| weight_decay | 0.01 |
| 初始 LR | 5e-5 |
| LR 衰减 | `lr * (0.85 ** (round - 2))` |
| eta_min_ratio | 0.1 |
| scheduler | CosineAnnealingLR |
| grad_clip | max_norm=1.0 |
| 随机种子 | 41-45 (N=5) |

---

## 3. 数据配置

| 参数 | 值 |
|:---|:---|
| 数据集 | v9_g1prime (FROZEN) |
| 训练集 | 46 samples |
| 评估集 | 97 samples |
| 增广策略 | A_fail (失败导向) |
| K 倍率 | 1.0× |
| 增广类型 | condition_variant, hint_variant, format_variant |
| Easy/Hard 变体数 | 2 |
| Medium 变体数 | 3 |

### 数据格式

```python
text = f"输入: {example['input_text']}\n输出: {example['target_text']}"
labels[labels == pad_token_id] = -100  # 忽略 padding
```

---

## 4. 评估配置

| 参数 | 值 |
|:---|:---|
| max_new_tokens | 100 |
| num_beams | 1 |
| do_sample | False |
| prompt | `f"输入: {item['input_text']}\n输出:"` |
| SMILES 提取 | V5 frozen (patterns_priority) |

---

## 5. 关键约束与禁令

### ⚠️ 必须遵守

1. **精度必须 bf16** — fp16 导致 Loss=NaN
2. **禁用 gradient checkpointing** — LoRA 兼容性问题
3. **禁用 FP16** — 已证明数值不稳定
4. **Recipe hash 必须匹配** — 防止配方漂移

### 🚫 禁止事项

| 禁止 | 原因 |
|:---|:---|
| fp16 训练 | NaN 梯度 (2026-11-18 诊断确认) |
| gradient checkpointing | cuDNN frontend 错误 |
| batch_size > 1 | 7B 显存限制 (40GB GPU) |
| 修改 target_modules | 已验证的最优配置 |
| 修改 LR/衰减 | 已冻结 |
| 训练数据变更 | v9_g1prime FROZEN |

---

## 6. 性能基准

| 指标 | A_fail (N=5) | A_rand (N=5) |
|:---|:---:|:---:|
| Peak mean±SD | **77.73±2.67%** | 11.13±0.04% |
| Final mean±SD | **75.26±5.15%** | 9.90±1.65% |
| Drawdown mean | 5.57% | 1.44% |
| 过线率 | **5/5** | 0/5 |

**最佳种子**: seed43 (Peak 81.44%, Final 80.41%)

---

## 7. 过线标准

| 指标 | 阈值 | 状态 |
|:---|:---|:---:|
| Peak ≥ 21.65% | ✅ 77.73% | PASS |
| Final ≥ 18.62% | ✅ 75.26% | PASS |
| Δ vs 1.5B M2 | +56pp | PASS |
| Δ vs A_rand | +66.6pp | PASS |

---

## 8. 脚本路径

| 文件 | 路径 |
|:---|:---|
| 主实验脚本 | `scripts/canonical/p2b_7b_experiment.py` |
| 批量运行脚本 | `scripts/canonical/p2b_batch_run.sh` |
| bf16 验证脚本 | `scripts/canonical/p2b_bf16_verify.py` |
| 诊断探针 | `scripts/canonical/p2b_diagnostic.py` |

---

## 9. 版本历史

| 日期 | 变更 | 原因 |
|:---|:---|:---|
| 2026-11-18 | fp16 → bf16 | NaN 梯度修复 |
| 2026-11-18 | 禁用 gradient checkpointing | cuDNN 兼容性问题 |
| 2026-11-18 | 配方冻结 | 防止漂移 |

---

## 10. 关联文档

- 完整报告: `docs/p2b_final_report.md`
- 裁决报告: `docs/p2b_verdict.md`
- V5 规范: `configs/canonical/v5_recipe_lock.md`
- Recipe hash: `configs/canonical/recipe_hash.txt`

---

**状态**: 🟢 已冻结  
**最后更新**: 2026-11-18  
**维护者**: RSI-MVE 实验组
