# RSI-MVE: Recursive Self-Improvement Minimal Viable Experiment

> **Status**: Phase 3 Complete — Canonical recipe frozen (2026-11-13)

## Quick Start

**唯一合法训练入口**:

```bash
python scripts/canonical/train_canonical.py \
  --config configs/canonical/v5_g1prime_canonical.yaml \
  --seed <SEED>
```

**G3/G4 实验**:

```bash
python scripts/canonical/g3_g4_canonical.py --experiment G3 --mode A_fail --seed <SEED>
python scripts/canonical/g3_g4_canonical.py --experiment G3 --mode A_rand --seed <SEED>
python scripts/canonical/g3_g4_canonical.py --experiment G4 --mode B_recur --seed <SEED>
python scripts/canonical/g3_g4_canonical.py --experiment G4 --mode B_single --seed <SEED>
```

## Canonical Configuration

| 参数 | V5 规范值 |
|------|----------|
| Precision | float32 |
| LR schedule | `5e-6 × 0.85^(r-2)` (decaying) |
| Batch size | 2 (direct) |
| Weight decay | 0.01 |
| Eta min ratio | 0.1 |
| Rounds | 7 |
| Augmentation | G1' prompt-only (5 types, English) |

**Recipe Hash**: `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3`

## Phase 3 Key Findings

| Claim | Result | Verdict |
|-------|--------|---------|
| A_fail ≈ A_rand (P1 G3) | **Δ = +13.0pp** | ❌ REFUTED |
| B_recur ≈ B_single (P1 G4) | Δ = +1.65pp | ⚠️ PARTIAL |
| Precision is root cause | **Protocol drift** | ❌ REFUTED |

**Root cause of P1 null results**: Three protocol deviations (D1/D2/D3) in the P1 training script masked true causal effects.

## Deprecated Scripts

The following scripts contain protocol drift and **MUST NOT BE RUN**:

- `scripts/_DEPRECATED_smoke_g1prime_train.py.bak` — D1/D2/D3 drift
- `scripts/_DEPRECATED_smoke_g1prime_v2.py.bak` — D1/D2/D3 drift

## Documentation

| Document | Path |
|----------|------|
| Phase 3 Report | `artifacts/canonical/Phase3_Canonical_Report.md` |
| Canonical Config | `configs/canonical/v5_g1prime_canonical.yaml` |
| Recipe Hash | `configs/canonical/recipe_hash.txt` |
| P0 Runbook | `../P0_immediate_runbook.md` |

## Directory Structure

```
RSI_MVE/
├── scripts/
│   ├── canonical/           # ← 唯一合法入口
│   │   ├── train_canonical.py
│   │   ├── g3_g4_canonical.py
│   │   └── ablation_study.py
│   └── _DEPRECATED_*.bak    # ← 禁止运行
├── configs/canonical/
│   ├── v5_g1prime_canonical.yaml
│   └── recipe_hash.txt
├── data/v9_g1prime/         # 无泄露数据集
└── artifacts/canonical/     # 所有规范实验产物
```

---

*Last updated: 2026-11-13*
