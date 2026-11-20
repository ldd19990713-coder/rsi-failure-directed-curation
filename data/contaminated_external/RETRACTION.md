# RETRACTION: Contaminated External Set (n=100)

**Status:** DEPRECATED — results on this set have been retracted.

**Date of retraction:** 2026-11-19

---

## What happened

During external evaluation (§6 of the paper), we discovered that the initially
reported "external" set (`external_eval_contaminated_n100.json`, n=100) had
**82 out of 100 samples with full reaction-key overlap** with the eval set.
Although sample IDs differed, the underlying
`(reactants, reagents, conditions) → product` tuples were duplicated.

This means the "external" set was **not truly external** — it contained
near-duplicates of evaluation samples, inflating the reported 63% accuracy.

## Audit procedure

1. Extract `(reactant_keys, product_key)` tuples from each split.
2. Compute intersection between `train∪eval` and the external set.
3. Full reaction key = canonical SMILES of `(reactants, reagents, conditions) → product`.
4. Result: **82/100 overlap** (fail).

The `neighbor_audit_report.json` file in this directory contains the detailed
audit results.

## Action taken

- **Retracted** the 63% accuracy claim reported on the contaminated set.
- **Rebuilt** a clean external set (n=31) with **zero reaction-key overlap**
  with `train∪eval` (see `../holdout_clean/`).
- **Re-ran** all external evaluations on the clean set:
  - Base model (zero-shot): 0.00%
  - Fine-tuned LoRA (seed 43): 0.00%
- Updated the paper (§6, §7.4, Appendix B) to reflect the retraction and
  the clean-set results.

## Lesson

ID-level deduplication alone is **insufficient** for external set validation.
Full reaction-key audit is necessary to prevent evaluation contamination from
masquerading as generalization. See §7.4 of the paper for a detailed
discussion.
