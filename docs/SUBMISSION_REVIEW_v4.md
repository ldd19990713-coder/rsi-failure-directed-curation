# RSI Paper v4 — Submission-Standard Review Report

**Review date:** 2026-11-20  
**Source:** `RSI_paper_v4_bundle.tar.gz` (Desktop)  
**Outcome:** Corrected in place → synced Desktop + `docs/paper/` (same filenames; tar refreshed)

---

## Verdict

**Technically ready for internal coauthor / venue-prep review after the fixes below.**  
Core claims (A_fail ≫ A_rand in-domain; negative controls; clean hold-out 0%; retraction) are consistent with locked P2b / M1–M3 / clean-external evidence.  
**Not yet camera-ready** for most venues: bibliography still thin on in-text citations; ASCII Fig.1 should become a vector figure; missing authorship/COI/ethics blocks if required by the target journal.

---

## Critical fixes applied (must-fix)

| Issue | Severity | Action |
|:---|:---|:---|
| Table 1 claimed A_rand final ≈2–3% | **Critical factual error** (confused with 1.5B G3 A_rand final ~2.47%) | Restored 7B finals: A_rand **9.90%±1.65%**, Δ_final **+65.36pp** |
| Qwen cite `arXiv:2502.13923` | **Wrong ID** | Corrected to **`arXiv:2412.15115`** |
| Multiple fabricated / misattributed refs (Hochreiter “Turing test”, Scellier ICML 2020, Lee et al. on 2303.10158, Welleck 2305.16239, blog/workshop placeholders, Settles initial “S.” / year) | **Integrity risk** | Replaced with verifiable set [1]–[15]; Zha et al. for data-centric survey; B. Settles 2009 |
| Emoji ✅/❌ in scope, tables, appendix | **Unprofessional for most journals** | Replaced with plain “In/Out of scope”, pass/fail, Demonstrated/Not observed |
| “Matched budget” vs Appendix C.3 nominal K | **Overclaim** | Aligned to *nominal* K-match + C.3 caveat |
| “Quality over quantity” wording | **Overclaim** | Softened to selection-under-budget |
| Appendix A `η_min=0.1` vs body `η_min/η_max=0.1` | **Ambiguous** | Unified to **η_min/η_max = 0.1** |
| EN paper bilingual Chinese subtitle | **Venue mismatch** for EN-only journals | Removed from EN; kept EN subtitle on ZH |
| Missing blank line before Future work | Style | Fixed (EN) |
| G3 *p*<0.01 presented as confirmatory | Mild overclaim | Labeled **exploratory; N=5** |

**SD note:** Paper Peak SD **2.79%** / Final SD **5.41%** match *sample* SD (ddof=1) from per-seed tables. Some lock docs list 2.67%/5.15% (likely population SD or rounding); **prefer paper figures**.

---

## Consistency checks (passed after edit)

- Recipe hash 7B / 1.5B unchanged and consistent footer ↔ §3 ↔ App. A  
- Main numbers: 77.73±2.79 / 11.13±0.46 / +66.60pp; seed table; seed43 asterisk  
- M3 −6.19pp; P1 77.32→70.10 (−7.22pp); hold-out n=31, 0%/0%; retract 82/100 → 63%  
- M1/M2 caveats (Full5 ≠ G3; nominal K; R2 peak for P2a) retained  
- max_length 256 present; three-layer → RQ0–4 funnel intact  
- Limitations: few-shot / n=31 / type-name overlap / skipped multi-seed external — adequate  

---

## Remaining submission gaps (not auto-fixed)

1. **In-text citations:** Related Work barely cites [4]–[15]; add pointers for SMILES, AL, curriculum, Molecular Transformer, data-centric survey, self-improve LLMs.  
2. **Figures:** Replace ASCII Fig.1 / E1 / E2 with PDF/TikZ/SVG; keep “nominal K-matched” caption.  
3. **Front matter:** Author list, affiliations, corresponding email, ORCID, funding, competing interests, ethics (data from public/internal corpus).  
4. **Abstract bold:** Many venues disallow heavy `**` markup—strip when converting to LaTeX.  
5. **Statistical language:** Prefer reporting exact paired *t* and CI for 7B peak Δ, or state “descriptive only; N=5” for main claim (sign consistency 5/5 already strong).  
6. **Supplementary:** Point to actual file names for n=31 SMILES table + scripts (currently “Supplementary Material” placeholder).  
7. **Venue format:** Convert MD → LaTeX/Word per template (Nature-style / ACS / NeurIPS differ sharply).  
8. **ZH↔EN parity:** Spot-check after further EN-only edits; both updated this pass for the critical items above.

---

## Claim-class judgment (for editors / yourself)

| Claim | Status |
|:---|:---|
| Failure-directed curation causes large in-domain gains vs random under lock | **Supported** |
| Hard rollback / naive scaling / local LR tweak fail as improvements | **Supported** |
| Cross-split exact-match generalization under clean key-disjoint hold-out | **Not observed (0%)** |
| Open-ended RSI / chemical rule learning | **Correctly denied** |
| Contaminated external 63% | **Correctly retracted** |

---

## Files updated

- `artifacts/RSI_paper_v4_bundle/RSI_paper_v4.md`
- `artifacts/RSI_paper_v4_bundle/RSI_paper_v4_zh.md`
- `docs/paper/RSI_paper_v4.md`, `RSI_paper_v4_zh.md`
- Desktop: `RSI_paper_v4.md`, `RSI_paper_v4_zh.md`, `RSI_paper_v4_bundle.tar.gz`
