# Failure-Directed Data Curation for Few-Shot Reaction Product Prediction

> Replication package for: Li, Wei, Luo & Yuan (2026). *"Failure-Directed Data
> Curation for Few-Shot Reaction Product Prediction: In-Domain Gains under
> Protocol Fidelity without Cross-Split Generalization."*

[![DOI](https://zenodo.org/badge/[DOI].svg)](https://doi.org/[DOI])

---

## Overview

This repository provides the **complete replication package** for our study on
bounded, data-centric self-refinement under a recursive self-improvement (RSI)
agenda. The central question: when does failure-directed data curation
(augmenting from model errors) produce verifiable gains, and where does it
fail?

### Key results (N=5 seeds, Qwen2.5-7B + LoRA, bf16)

| Metric | A\_fail (ours) | A\_rand (control) | Δ |
| :----- | :---: | :---: | :---: |
| Peak accuracy | **77.73% ± 2.79%** | 11.13% ± 0.46% | **+66.60pp** |
| Final accuracy | 75.26% ± 5.41% | 9.90% ± 1.65% | +65.36pp |
| Sign consistency | 5/5 seeds same direction |

**Negative controls:**
- M3 hard rollback: peak −6.19pp (premature lock harmful)
- P2a naive scaling: early-peak collapse (distribution shift ≠ curation)
- P1 LR-decay tweak: peak −7.22pp (default recipe robust)

**Hold-out (n=31, zero reaction-key overlap with train∪eval):**
- Base model (zero-shot): 0.00%
- Fine-tuned LoRA (seed 43): 0.00%
- → Cross-split exact-match generalization **not observed**

**Retraction:** A previously reported 63% external accuracy was retracted
after discovering 82/100 samples had full reaction-key overlap with the eval
set (see `data/contaminated_external/RETRACTION.md`).

---

## Quick start

```bash
# 1. Environment
conda create -n rsi-fc python=3.11 -y && conda activate rsi-fc
pip install -r requirements.txt

# 2. Download base model (≈15 GB)
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ./models/Qwen2.5-7B

# 3. Verify data integrity (reaction-key disjointness)
python evaluation/key_disjoint_check.py \
    --train data/v9_g1prime/train.json \
    --eval data/v9_g1prime/eval.json \
    --holdout data/holdout_clean/holdout_n31.json

# 4. Three-level leakage audit (train/eval/holdout)
python evaluation/leakage_audit.py \
    --data data/v9_g1prime \
    --holdout data/holdout_clean/holdout_n31.json

# 5. Reproduce primary result (seed 41, A_fail, 7B) — needs GPU + HF model
python scripts/train_lora.py \
    --config configs/recipe_7b.yaml \
    --strategy a_fail \
    --seed 41 \
    --output_dir runs/seed41_afail

# 6. Exact-SMILES evaluate (after you write predictions JSON)
python scripts/evaluate.py \
    --split data/v9_g1prime/eval.json \
    --preds runs/seed41_afail/preds.json

# 7. Mechanism validation (1.5B, M2 dose-response)
python scripts/mechanism/m2_dose_response.py
```

> Audits (steps 3–4) run with only the stdlib + JSON data. Training steps need a CUDA GPU and a download of `Qwen/Qwen2.5-7B` / `Qwen/Qwen2.5-1.5B`.

---

## Repository structure

```
rsi-failure-directed-curation/
├── README.md
├── LICENSE                          (MIT)
├── CITATION.cff
├── requirements.txt
├── configs/
│   ├── v5_g1prime_canonical.yaml    (locked hyperparameters)
│   ├── recipe_hash.txt              (SHA-256 of canonical recipe)
│   └── p2b_7b_lock.md               (7B recipe freeze protocol)
├── data/
│   ├── v9_g1prime/                  (46 train, 97 eval)
│   │   ├── train.json
│   │   ├── eval.json
│   │   └── metadata.json
│   ├── holdout_clean/               (n=31, reaction-key disjoint)
│   │   ├── holdout_n31.json
│   │   ├── audit_report.json
│   │   └── manifest.json
│   └── contaminated_external/       (DEPRECATED — see RETRACTION.md)
│       ├── external_eval_contaminated_n100.json
│       ├── RETRACTION.md
│       └── neighbor_audit_report.json
├── scripts/
│   ├── canonical/                   (primary training & evaluation)
│   │   ├── train_canonical.py
│   │   ├── g3_canonical.py
│   │   ├── g3_g4_canonical.py
│   │   ├── p2b_7b_experiment.py
│   │   ├── p2p_*.py                 (holdout eval, base + LoRA)
│   │   └── p2_external_eval.py
│   ├── mechanism/                   (Appendix C experiments)
│   │   ├── m1_leave_one_out.py
│   │   ├── m1_analysis.py
│   │   ├── m2_dose_response.py
│   │   └── m3_rollback.py
│   ├── evaluation/                  (metrics, stats)
│   │   ├── evaluate_all_models.py
│   │   ├── statistical_analysis_v5.py
│   │   └── ...
│   └── data/                        (preparation, dedup)
│       ├── prepare_data_100_v2.py
│       ├── deduplicate_data.py
│       ├── scan_near_duplicates.py
│       └── ...
├── evaluation/
│   ├── leakage_audit.py
│   └── key_disjoint_check.py
├── figures/                         (paper figures — source + rendered)
│   ├── fig1_method_pipeline.mmd
│   ├── render_fig1.sh
│   ├── figE1_p2a_scaling.{pdf,svg,png}
│   ├── figE2_m3_rollback.{pdf,svg,png}
│   └── render_negative_trajectories.py
├── paper/
│   ├── RSI_paper_v7.md             (English)
│   └── RSI_paper_v7_zh.md          (Chinese)
├── docs/                            (verdicts, reports, protocols)
└── OPEN_SOURCE_PLAN.md              (open-source disclosure plan)
```

---

## Data

| Split | Samples | Description |
| :---- | ------: | :---------- |
| `train` | 46 | 23 reaction types, 1–4 samples per type |
| `eval` | 97 | 21 reaction types, zero ID/key overlap with train |
| `holdout_clean` (n=31) | 31 | Zero reaction-key overlap with train∪eval; 18 type labels (14 shared name, 4 entirely new) |
| `contaminated_external` (n=100) | 100 | **DEPRECATED** — 82/100 had full reaction-key overlap with eval |

Each record in the JSON files contains:
- `id`: unique sample identifier
- `input_text`: formatted prompt (reactant SMILES + conditions)
- `target_text`: ground-truth product SMILES
- `product_smiles`: canonical product SMILES
- `reaction_type`: named reaction class
- `reaction_key`: canonical (reactants, reagents, conditions) → product tuple (for leakage auditing)
- `reactants_smiles`, `difficulty`, etc.

---

## Recipe hash

We freeze all hyperparameters after initial exploration and report a SHA-256
hash for reproducibility:

| Model | Hash |
| :---- | :--- |
| **7B** | `9e3788dd7a2eb89d66a1b403d5c5b7c425d583adf7c799c35b761b57fbf7f809` |
| **1.5B** | `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3` |

See `configs/v5_g1prime_canonical.yaml` and `configs/p2b_7b_lock.md` for the
full locked specification.

---

## Figures

Figures in the paper are provided in vector format (PDF/SVG). Source code is
included for reproducibility:

| Figure | Source | Render command |
| :----- | :----- | :------------- |
| Fig 1 (method pipeline) | `figures/fig1_method_pipeline.mmd` | `bash figures/render_fig1.sh` |
| Fig E1 (P2a scaling) | `figures/render_negative_trajectories.py` | `python figures/render_negative_trajectories.py` |
| Fig E2 (M3 rollback) | `figures/render_negative_trajectories.py` | `python figures/render_negative_trajectories.py` |

Figure 1 requires `mmdc` (`npm install -g @mermaid-js/mermaid-cli`). Figures
E1/E2 require only `matplotlib` and `numpy`.

---

## Citation

```bibtex
@article{li2026failure,
  title={Failure-Directed Data Curation for Few-Shot Reaction Product Prediction:
         In-Domain Gains under Protocol Fidelity without Cross-Split Generalization},
  author={Li, Dong and Wei, Yixiong and Luo, Shi and Yuan, Yingjun},
  year={2026},
  doi={[DOI]}
}
```

---

## License

This replication package is released under the [MIT License](LICENSE).

The base model (Qwen2.5-7B / Qwen2.5-1.5B) is publicly available on Hugging
Face and subject to its own license terms.

---

## Contact

Dong Li (lidong@zhejianglab.org) — corresponding author.
