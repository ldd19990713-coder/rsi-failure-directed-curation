# Public release checklist (executed 2026-11-20)

## Done locally

- [x] MIT LICENSE + CITATION.cff + README
- [x] Data: `v9_g1prime` (46/97), `holdout_clean` (n=31), contaminated retraction pack
- [x] Added `reaction_key` to train/eval for auditability
- [x] Rewrote `evaluation/leakage_audit.py` (CLI, no internal paths)
- [x] Updated `evaluation/key_disjoint_check.py` (derives keys if missing)
- [x] Audit smoke: **PASS** (train↔eval key overlap 0; holdout key overlap 0)
- [x] Sanitized `/home/scientist/...` absolute paths → repo-root walk / HF hub ids
- [x] Public configs: `configs/recipe_7b.yaml`, `recipe_1.5b.yaml`, `recipe_hashes.json`
- [x] Public entrypoints: `scripts/train_lora.py`, `scripts/evaluate.py`
- [x] Figures under `figures/`; paper under `paper/`
- [x] Local `git init -b main`

## Pending (requires GitHub auth)

- [ ] `gh auth login` (device code) → create **public** repo
- [ ] Push `main` + tag `v1.0.0` release
- [ ] Enable Zenodo↔GitHub → obtain DOI
- [ ] Replace `https://github.com/ldd19990713-coder/rsi-failure-directed-curation` / `[DOI]` in README, CITATION.cff, `paper/RSI_paper_v7*.md`

## Notes

- Product-SMILES overlap between holdout and train/eval can be nonzero; **reaction-key** overlap is the hard gate (paper protocol).
- Legacy runners still expect GPU + HF download; prefer `scripts/train_lora.py` / canonical scripts after setting `RSI_ROOT`.
- LoRA adapters are **not** shipped (size); available on request as stated in the paper.
