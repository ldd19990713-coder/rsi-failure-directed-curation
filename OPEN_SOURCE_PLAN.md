# 代码与数据公开方案（Open-Source Disclosure Plan）

> 本文档为论文 V7 中"Data and Code Availability"章节的执行配套方案。
> 作者按步骤完成公开后，将所得 URL / DOI 回填至论文的相应占位符 `[REPO_URL]` / `[DOI]` 即可。

---

## 1. 推荐仓库结构

```
rsi-failure-directed-curation/
├── README.md                    ← 必须
├── LICENSE                      ← 必须（推荐 MIT 或 Apache-2.0）
├── CITATION.cff                 ← 必须（Zenodo / GitHub 引用元数据）
├── .gitignore                   ← 必须
├── requirements.txt             ← Python 依赖
├── pyproject.toml               ← 可选（PEP 621 项目元数据）
│
├── configs/                     ← 配方锁定文件
│   ├── recipe_7b.yaml           ← 7B 超参数
│   ├── recipe_1.5b.yaml         ← 1.5B 超参数
│   └── recipe_hashes.json       ← 配方哈希
│
├── data/
│   ├── v9_g1prime/              ← 主数据集划分
│   │   ├── train.jsonl          ← 46 个训练反应
│   │   ├── eval.jsonl           ← 97 个评测反应
│   │   └── metadata.json        ← 划分元数据（ID/反应键/类型）
│   ├── holdout_clean/           ← 干净外集
│   │   ├── holdout_n31.jsonl    ← 31 个样本（含完整 SMILES + 反应类型 + 审计元数据）
│   │   └── audit_report.json    ← 反应键审计结果
│   └── contaminated_external/   ← 已弃用的污染集（注明废弃原因）
│       └── retraction_note.md
│
├── scripts/
│   ├── train_lora.py            ← 主训练入口（LoRA fine-tune）
│   ├── evaluate.py              ← 评测脚本（exact SMILES match）
│   ├── failure_augment.py       ← 失败定向增广（A_fail）
│   ├── random_augment.py        ← 随机增广（A_rand）
│   ├── m1_leave_one_type.py     ← 附录 C.2 留一类型实验
│   ├── m2_dose_response.py      ← 附录 C.3 剂量-反应曲线
│   ├── m3_rollback.py           ← §5.1 硬回滚实验
│   └── p2a_naive_scaling.py     ← §5.2 朴素扩容实验
│
├── evaluation/
│   ├── leakage_audit.py         ← 泄露审计（ID / 产物 / 反应键三级）
│   ├── key_disjoint_check.py    ← 干净外集反应键不相交验证
│   └── tests/                   ← 评测脚本单元测试
│       └── test_evaluate.py
│
├── logs/                        ← 代表性训练日志（可选）
│   ├── seed41_afail.log
│   └── seed43_afail.log
│
├── figures/                     ← 论文矢量图源码
│   ├── fig1_method_pipeline.mmd
│   ├── figE1_p2a_scaling.pdf
│   ├── figE2_m3_rollback.pdf
│   ├── render_fig1.sh
│   └── render_negative_trajectories.py
│
└── paper/                       ← 论文源码（可选，鼓励）
    ├── RSI_paper_v7.md
    └── RSI_paper_v7_zh.md
```

---

## 2. 必备文件模板

### 2.1 README.md

```markdown
# Failure-Directed Data Curation for Few-Shot Reaction Product Prediction

> Replication package for the paper "Failure-Directed Data Curation for
> Few-Shot Reaction Product Prediction: In-Domain Gains under Protocol
> Fidelity without Cross-Split Generalization" (Li et al., 2026).

## Overview

This repository provides the complete replication package for our study on
bounded, data-centric self-refinement under a recursive self-improvement
(RSI) agenda. Key results:

- **A_fail vs A_rand (in-domain):** 77.73% ± 2.79% vs 11.13% ± 0.46%
  (+66.60pp, N=5 seeds, 5/5 same direction)
- **Negative controls:** M3 hard rollback (−6.19pp), P2a naive scaling
  (early-peak collapse), P1 LR-decay tweak (−7.22pp)
- **Hold-out (n=31, reaction-key-disjoint):** 0% / 0% for base + LoRA

## Quick start

```bash
# 1. Environment
conda create -n rsi python=3.11 -y && conda activate rsi
pip install -r requirements.txt

# 2. Download base model
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ./models/Qwen2.5-7B

# 3. Reproduce primary result (seed 41, A_fail)
python scripts/train_lora.py \
    --config configs/recipe_7b.yaml \
    --data data/v9_g1prime \
    --strategy a_fail \
    --seed 41 \
    --output_dir runs/seed41_afail

# 4. Evaluate
python scripts/evaluate.py \
    --adapter runs/seed41_afail \
    --split eval

# 5. Leakage audit
python evaluation/leakage_audit.py --data data/v9_g1prime
```

## Data

| Split | Samples | Description |
| :---- | ------: | :---------- |
| train | 46 | 23 reaction types, 1–4 samples per type |
| eval | 97 | 21 reaction types |
| holdout_clean (n=31) | 31 | Zero reaction-key overlap with train∪eval |

The clean hold-out set is provided in `data/holdout_clean/holdout_n31.jsonl`
with full SMILES, reaction types, and audit metadata.

## Recipe hash

- **7B:**  `9e3788dd7a2eb89d66a1b403d5c5b7c425d583adf7c799c35b761b57fbf7f809`
- **1.5B:** `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3`

## Citation

```bibtex
@article{li2026failure,
  title={Failure-Directed Data Curation for Few-Shot Reaction Product Prediction:
         In-Domain Gains under Protocol Fidelity without Cross-Split Generalization},
  author={Li, Dong and Wei, Yixiong and Luo, Shi and Yuan, Yingjun},
  year={2026},
  url={https://doi.org/[DOI]}
}
```

## License

[MIT](LICENSE)
```

### 2.2 CITATION.cff

```yaml
cff-version: 1.2.0
title: "Failure-Directed Data Curation for Few-Shot Reaction Product Prediction"
authors:
  - family-names: Li
    given-names: Dong
    orcid: https://orcid.org/0009-0009-8133-5796
    affiliation: "Zhejiang Lab"
  - family-names: Wei
    given-names: Yixiong
    orcid: https://orcid.org/0009-0006-0747-2897
    affiliation: "Zhejiang Lab"
  - family-names: Luo
    given-names: Shi
    orcid: https://orcid.org/0009-0002-1424-4105
    affiliation: "Zhejiang Lab"
  - family-names: Yuan
    given-names: Yingjun
    orcid: https://orcid.org/0009-0004-3595-3247
    affiliation: "Zhejiang Lab"
message: "If you use this software, please cite it using the metadata from this file."
repository-code: "[REPO_URL]"
url: "[DOI_URL]"
license: MIT
version: 1.0.0
date-released: 2026-XX-XX
```

### 2.3 LICENSE（推荐 MIT）

```
MIT License

Copyright (c) 2026 Zhejiang Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 2.4 .gitignore

```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# Models / large binaries
models/
*.bin
*.safetensors
*.pt
*.ckpt

# Logs / runs
runs/
wandb/
outputs/
*.log

# IDE
.vscode/
.idea/
.DS_Store
```

---

## 3. 发布流程

### Step 3.1 在 GitHub 创建公开仓库

```bash
# 本地初始化
cd /path/to/rsi-failure-directed-curation
git init
git add .
git commit -m "Initial commit: replication package"

# 推到 GitHub
gh repo create rsi-failure-directed-curation --public --source=. --push
# 或手动：
# git remote add origin git@github.com:<YOUR_USERNAME>/rsi-failure-directed-curation.git
# git branch -M main
# git push -u origin main
```

> 仓库命名建议：`rsi-failure-directed-curation` 或 `failure-directed-curation-few-shot-reaction`

### Step 3.2 与 Zenodo 集成（自动获取 DOI）

1. 登录 <https://zenodo.org>（可用 GitHub 账号直接登录）
2. 进入 **GitHub** 页面：<https://zenodo.org/account/settings/github/>
3. 点击 **Select repositories**，勾选 `rsi-failure-directed-curation`
4. 此后每次在 GitHub 上发布 **Release**，Zenodo 会自动归档并分配 DOI

### Step 3.3 在 GitHub 创建首个 Release

```bash
gh release create v1.0.0 \
    --title "v1.0.0 — Paper submission release" \
    --notes "Initial public release accompanying the paper submission."
```

或在 GitHub Web UI 上：
- **Releases** → **Draft a new release**
- Tag: `v1.0.0`
- Title: `v1.0.0 — Paper submission release`
- 勾选 "Create a new tag on publish"

→ Zenodo 会自动归档 → 获得 DOI（如 `10.5281/zenodo.XXXXXXX`）

### Step 3.4 在 Zenodo 完善元数据

- **Title**: 与论文一致
- **Authors**: Dong Li, Yixiong Wei, Shi Luo, Yingjun Yuan（含 ORCID）
- **Affiliation**: Zhejiang Lab
- **Related identifiers**: 指向论文（如已有 preprint DOI）
- **License**: MIT
- **Keywords**: recursive self-improvement, data-centric AI, failure-directed curation, few-shot learning, chemical reaction prediction

### Step 3.5 回填到论文

将以下占位符替换为实际值：

| 占位符 | 替换为 |
|:------|:------|
| `[REPO_URL]` | `https://github.com/<USER>/rsi-failure-directed-curation` |
| `[DOI]` | `10.5281/zenodo.XXXXXXX` |
| `[DOI_URL]` | `https://doi.org/10.5281/zenodo.XXXXXXX` |

---

## 4. 回填后的论文章节

### Data and Code Availability（EN）

```
The clean external evaluation set (n=31), recipe hashes, evaluation scripts,
and the full replication package are publicly available at [REPO_URL]
(archived DOI: [DOI]). The base model (Qwen2.5-7B) is publicly available on
Hugging Face.
```

### 数据与代码可用性（ZH）

```
干净外集（n=31）、配方哈希、评测脚本及完整复现代码已公开至 [REPO_URL]
（归档 DOI：[DOI]）。基座模型（Qwen2.5-7B）可在 Hugging Face 公开获取。
```

---

## 5. 核查清单

公开前请确认以下事项：

- [ ] 仓库中**不包含**任何：
  - 模型权重文件（> 50MB，请使用 Git-LFS 或不上传）
  - 私有 API 密钥 / 凭据
  - 未脱敏的原始数据（如有第三方限制）
- [ ] README 中的 Quick start 已在新环境中复现通过
- [ ] `holdout_n31.jsonl` 已含完整 SMILES / 反应类型 / 审计元数据
- [ ] `leakage_audit.py` 可独立运行并通过
- [ ] LICENSE 与 CITATION.cff 已就位
- [ ] 配方哈希与论文一致（7B: `9e3788dd...`, 1.5B: `2a965c39...`）
- [ ] 论文数字（77.73±2.79 / 11.13±0.46 / +66.60pp）与脚本可复现

---

## 6. 时间估算

| 步骤 | 预计耗时 |
|:-----|:---------|
| 整理代码 + 写 README | 2–4 小时 |
| GitHub 创建仓库 + 初推 | 30 分钟 |
| Zenodo 集成 + 首次 Release | 30 分钟 |
| 回填论文 | 10 分钟 |
| **总计** | **半天内可完成** |
