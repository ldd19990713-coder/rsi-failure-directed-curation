# Failure-Directed Data Curation for Few-Shot Reaction Product Prediction: In-Domain Gains under Protocol Fidelity without Cross-Split Generalization

***

**Authors:** Dong Li¹, Yixiong Wei¹, Shi Luo¹, Yingjun Yuan¹

**Affiliation:**
¹ Center for Earth and Space Science Computing, Zhejiang Lab, Hangzhou, China

**Corresponding author:** Dong Li (lidong@zhejianglab.org)

**ORCID:**
Dong Li: 0009-0009-8133-5796
Yixiong Wei: 0009-0006-0747-2897
Shi Luo: 0009-0002-1424-4105
Yingjun Yuan: 0009-0004-3595-3247

***

## Abstract

Research on recursive self-improvement (RSI) asks when closed-loop model updates yield verifiable gains—and where they fail. A durable scientific intuition—shared by empirical research practice and everyday problem solving—is that lessons drawn from failure are often more informative than lessons drawn from success, because errors localize the hypothesis space that still needs revision. We instantiate this intuition, within an RSI agenda, as bounded, data-centric self-refinement: preferentially repairing what the model gets wrong, rather than asserting open-ended RSI. Few-shot reaction product prediction under exact SMILES matching is a stringent testbed: supervision is scarce, and protocol drift or contaminated evaluation can manufacture false gains. Under a leakage-controlled split (46 train / 97 eval), we test whether failure-directed data curation—augmenting from model errors—outperforms matched random augmentation, whether recursive scheduling or naive scaling helps, and whether in-domain gains transfer to key-disjoint hold-outs. With a frozen Qwen2.5-7B + LoRA (bf16) recipe over N=5 seeds, failure-directed augmentation (A\_fail) reaches 77.73%±2.79% peak accuracy versus 11.13%±0.46% for matched random augmentation (A\_rand; Δ = +66.60pp). Hard rollback harms peak by locking early checkpoints; naively merging distribution-shifted extras induces early-peak collapse; a local LR-decay tweak fails a pre-registered stability gate. On a held-out set with zero reaction-key overlap with train∪eval (n=31; 18 type labels, of which 14 appear as names in training but with different instances), both the base model and the fine-tuned LoRA obtain 0% exact-match accuracy. We also retract a previously reported external score inflated by eval-key duplication. We conclude that failure-directed curation yields large, reproducible in-domain gains that do not transfer under our held-out protocol, clarifying the conditions and boundaries of data-centric self-refinement—not validating open-ended RSI or cross-split chemical generalization.

**Keywords:** recursive self-improvement, data-centric AI, failure-directed curation, few-shot learning, chemical reaction prediction, evaluation integrity

***

## 1\. Introduction

### 1.1 From RSI to Bounded Self-Refinement: A Three-Layer Problem Stack

This paper is situated in research on **recursive self-improvement (RSI)**: when iterative model updates can produce *verifiable* performance gains across rounds, and how to control catastrophic drawdown and overfitting. We treat "open-ended / continuous RSI" as a **long-horizon evaluation horizon**, not as a claim of this work. Instead, we adopt a three-layer problem stack that narrows from domain-level RSI questions to falsifiable experiments.

**Layer I — Domain question (Why).** Under *bounded* settings, when (and how) can recursive fine-tuning produce **auditable, round-to-round improvement** while controlling catastrophic forgetting and overfitting? Clarifying these **conditions, boundaries, and infeasible regions** is a prerequisite for any stronger RSI narrative.

**Layer II — Scene and mechanism (Where / What).** We specialize Layer I to a **data-centric** closed loop in **few-shot chemical reaction product prediction** (exact SMILES): (i) **failure-directed augmentation** as the data policy (fault → data repair); (ii) **multi-round weight updates** (LoRA fine-tuning) as the execution layer. This choice is deliberate. In scientific practice as in daily life, **learning from failure is often more consequential than rehearsing success**: successes mainly confirm what already works, whereas failures mark the frontier where the current model of the world is incomplete. Operationally, we therefore treat model errors as the primary curriculum signal—not as noise to be averaged away by random extra data. The goal is to identify the **causal contribution** of this failure-centric loop relative to matched controls, and the failure modes induced by forgetting, overfitting, and weak evaluation protocol. We do **not** study meta-level "improving the improver."

**Layer III — Operational focus (How).** Layer II is operationalized under a **leakage-controlled, fixed split** with exact-SMILES metrics (peak / final / drawdown), recipe locking, and pre-registered contrasts: failure-directed vs random augmentation; recursive scheduling / scaling interventions vs defaults; in-split gains vs key-disjoint hold-out. Only under this discipline do we admit claims of "improvement."

**Funnel.** Layer I motivates *why* closed-loop gains must be bounded and auditable; Layer II chooses *which* mechanism (data-centric failure repair in chemistry) to test; Layer III yields the concrete RQs below. Positive in-domain effects answer Layer II partially; hold-out zeros and negative controls delineate Layer I boundaries.

### 1.2 Motivation for the Chemistry Testbed

Exact SMILES matching is unforgiving, supervision is scarce (here: 46 train / 97 eval), and evaluation contamination can masquerade as generalization. Strong "self-improving" claims in the broader RSI literature often rest on uncontrolled protocol drift or contaminated evaluation. Before claiming generalization or continuous self-improvement, one must establish: (1) that an intervention causes reproducible improvement under a **fixed** protocol, and (2) that improvement transfers to **genuinely key-disjoint** data. We address both for **failure-directed data curation**.

### 1.3 Research Questions (Layer III Instantiation)

**RQ0 (bridges Layers I–III):** Under leakage-controlled few-shot reaction product prediction, what closed-loop intervention yields reproducible **bounded** improvement, and where are its boundaries?

**RQ1 (data policy):** Does failure-directed augmentation outperform budget-matched random augmentation?

**RQ2 (execution / scheduling):** Can recursive scheduling interventions (hard rollback) or naive same-pool scaling further improve performance?

**RQ3 (Layer I boundary):** Do in-domain gains transfer to a hold-out set with zero reaction-key overlap with train∪eval?

**RQ4 (evaluation integrity):** How do protocol drift and evaluation contamination produce false negatives and false positives?

### 1.4 Scope and Positioning

This work studies **bounded data-centric self-refinement** within an RSI research agenda, explicitly positioned as:

* **In scope:** a controlled empirical study of failure-directed curation (Layer II mechanism)
* **In scope:** a protocol-fidelity investigation with leakage audit (Layer III discipline)
* **In scope:** a transparent report of negative results and retracted claims (Layer I boundaries)
* **Out of scope:** validation of open-ended / continuous RSI
* **Out of scope:** demonstration of reliable cross-split chemical generalization
* **Out of scope:** a production-ready system
* **Out of scope:** meta-level improver self-modification

### 1.5 Contributions

1. **Three-layer framing** (§1): Connect RSI-level questions to a falsifiable data-centric chemistry loop
2. **Protocol method** (§3): Leakage-controlled splits, recipe locking, and audit standards (RQ4)
3. **Main effect** (§4): A\_fail ≫ A\_rand (+66.60pp, N=5, RQ1)
4. **Negative controls** (§5): M3 rollback, P2a scaling, P1 LR decay all fail (RQ2)
5. **Hold-out zero generalization** (§6): 0% on clean external set (RQ3)
6. **Scientific integrity** (Appendix B): Retraction of contaminated external score (RQ4)

***

## 2\. Related Work

### 2.1 Few-Shot Chemical Reaction Prediction

Reaction product prediction in SMILES format requires exact string matching, making few-shot settings particularly challenging. Prior work on molecular representation learning [11, 12] and neural reaction prediction [13, 14] has made progress on larger datasets, but exact-match few-shot prediction in chemistry remains underexplored, with high variance and susceptibility to evaluation contamination. Computer-aided synthesis planning [15] further highlights the gap between classification-style benchmarks and the exact-SMILES setting studied here.

Large-scale reaction corpora such as USPTO have enabled strong sequence-to-sequence and graph baselines — augmented NLP Transformers reach top-5 ≈ 85% on USPTO-50K [17], and Graph2SMILES reports top-1 ≈ 55% on the same benchmark by combining Transformer decoding with permutation-invariant graph encoders [18] — but these settings assume tens of thousands of paired examples, not the few-shot regime of the present work. Recent chemical LLMs (ChemLLM [16], ChemDFM [19]) demonstrate that domain-specific pretraining and instruction tuning can close part of the gap on chemistry question-answering and reaction tasks, yet their evaluations target multi-shot benchmarks (e.g., ChemBench [16]) rather than exact-match few-shot product prediction. Comprehensive retrosynthesis surveys [20] likewise document that most ML methods require abundant paired data, leaving the few-shot product-prediction setting underexplored.

**Connection to our work:** The difficulty of exact-match few-shot prediction motivates our focus on *how* data are selected under a fixed budget (failure-directed vs random), and helps contextualize the 0% hold-out result.

### 2.2 Hard Example Mining and Failure-Driven Learning

Beyond ML heuristics, a broader epistemic stance holds that progress is often driven more by diagnosing failure than by repeating success—errors constrain which theories remain viable. Active learning [5] and curriculum learning [6] operationalize this view computationally: examples near the decision boundary, or on which the model currently fails, provide disproportionate information. Recent work on LLM self-improvement [9] echoes this theme but typically targets larger-scale settings. In few-shot settings, random augmentation dilutes signal; failure-directed selection concentrates revision where the predictor is still wrong.

**Connection to our work:** RQ1 directly tests this hypothesis: A\_fail vs A\_rand isolates the effect of selection strategy under matched data budget—i.e., whether "learning from mistakes" outperforms "learning from more of the same" when budgets are equal.

### 2.3 Data-Centric vs Algorithm-Centric Approaches

The data-centric AI movement [7, 8] argues that data quality and selection, not model architecture alone, are primary levers in many practical settings. Our work extends this to the few-shot regime: with only 46 training examples, *which* examples are curated under a matched budget matters more than undifferentiated volume.

**Connection to our work:** RQ2 negative controls (M3, P2a) demonstrate that naive scaling without curation is ineffective or harmful.

### 2.4 Recursive Self-Improvement (RSI) and Bounded Self-Refinement

RSI literature [1] discusses open-ended capability growth via self-generated training signals. Strong claims require at least: (1) reproducible improvement under a fixed protocol, (2) transfer beyond the training distribution, and (3) evaluation integrity—a concern also raised in broader transfer-learning evaluations [10]. Our **Layer I** question inherits this agenda but restricts the claim class to **bounded self-refinement**: auditable in-domain gains with explicit negative boundaries. We show that (1) holds for failure-directed curation under recipe lock, (2) fails under key-disjoint hold-out, and (3) is easily violated without reaction-key audit—hence continuous RSI is **not** established here.

**Connection to our work:** The three-layer stack (§1.1) maps RSI-level questions onto a data-centric chemistry loop; experiments test mechanisms and boundaries rather than asserting open-ended RSI.

***

## 3\. Problem Setup and Protocol Fidelity

### 3.1 Task Definition

**Task:** Given reactant SMILES and reaction conditions (type, reagents, solvent, temperature, time), predict the major product SMILES.

**Evaluation:** Exact string match after canonical SMILES normalization. No partial credit.

**Dataset split (v9\_g1prime):**

* Training: 46 reactions (23 types, 1–4 samples per type)
* Evaluation: 97 reactions (21 types)
* ID overlap: 0
* Full reaction key overlap: 0

### 3.2 Protocol Fidelity Measures

#### 3.2.1 Recipe Locking

We freeze all hyperparameters after initial exploration and report a recipe hash for reproducibility:

| Component | Value |
| :-------- | :---- |
| Base model | Qwen2.5-7B [2] |
| Precision | bfloat16 (fp16 forbidden — causes NaN) |
| LoRA rank | r=16, α=32 [3] |
| LoRA dropout | 0.05 |
| Target modules | q\_proj, v\_proj, k\_proj, o\_proj, gate\_proj, up\_proj, down\_proj |
| Learning rate | 5e-5; per-round decay `0.85^(r-2)` with CosineAnnealing (η\_min/η\_max = 0.1) |
| Batch size | 1 (40GB GPU constraint; no gradient accumulation) |
| Training rounds | 7 (3 epochs/round) |
| Augmentation | A\_fail, K=1.0×; prompt variants (condition/hint/format slice) |
| **Recipe hash** | `9e3788dd7a2eb89d66a1b403d5c5b7c425d583adf7c799c35b761b57fbf7f809` |

#### 3.2.2 Leakage Audit Standards

Before any experiment, we verify:

1. **ID overlap:** No shared IDs between train/eval/external
2. **Product overlap:** Check for shared product SMILES (non-leakage if different reactions)
3. **Full reaction key overlap:** Check for shared (reactants → product) tuples — **must be 0**

#### 3.2.3 Primary Metrics

* **Peak accuracy:** Maximum accuracy across training rounds
* **Final accuracy:** Accuracy at the last training round (round 7)
* **Drawdown:** Peak − post-peak minimum (catastrophic forgetting proxy)
* **Reported as:** Mean ± SD across N=5 seeds (41–45)

### 3.3 Method Overview

Figure 1 provides a visual summary of our method pipeline: from the three-layer RSI problem stack, through the failure-directed curation loop, to the evaluation gates that separate in-domain gains from generalization probes.

![Figure 1. Method pipeline](figures/fig1_method_pipeline.pdf)

**Figure 1.** Method pipeline: The three-layer RSI problem stack (top) motivates the failure-directed curation loop (middle), which is evaluated through three gates—In-Split (RQ1), Negative Controls (RQ2), and Hold-Out (RQ3)—leading to the final verdict of bounded in-domain refinement without cross-split transfer. "Nominal K-matched" denotes the intended equal-budget control; residual per-round count imbalance is discussed in Appendix C.3. *Vector source: `figures/fig1_method_pipeline.mmd`; render with `bash figures/render_fig1.sh` (requires `mmdc`).*

### 3.4 Data Integrity: Contaminated External Set Retraction

During external evaluation, we discovered that the initially reported "external" set (n=100) had **82/100 samples with full reaction key overlap** with the eval set. Although IDs differed, the underlying (reactants → product) tuples were duplicated.

**Action taken:**

* Retracted the 63% accuracy claim
* Rebuilt clean external set (n=31) with **zero reaction-key overlap** with train∪eval
* Re-ran all external evaluations on clean set

This incident underscores RQ4: evaluation contamination can produce false positives for generalization claims.

***

## 4\. Main Result: Failure\-Directed ≫ Random \(RQ1\)

### 4.1 Experimental Design

**Hypothesis:** If the information bottleneck lies in model error regions, then targeted augmentation of failure cases should outperform matched random augmentation.

**Setup:**

* Model: Qwen2.5-7B + LoRA (bf16)
* Seeds: 41, 42, 43, 44, 45
* Comparison: A\_fail (failure-directed) vs A\_rand (random sampling from same pool)
* Data budget: *Nominally* K-matched across conditions (residual per-round count imbalance noted in Appendix C.3)

### 4.2 Results

**Table 1: In-Split Results (N=5, Mean ± SD; sample SD)**

| Method | Peak Accuracy | Final Accuracy |
| :----- | :-----------: | :------------: |
| **A\_fail (ours)** | **77.73% ± 2.79%** | **75.26% ± 5.41%** |
| A\_rand (baseline) | 11.13% ± 0.46% | 9.90% ± 1.65% |
| **Δ** | **+66.60pp** | **+65.36pp** |

Primary claim uses peak accuracy (pre-registered peak contrast). Final-round Δ is reported for completeness; A\_rand remains near \~10% throughout, so the gap is not an artifact of peak-picking alone.

**Table 2: Per-Seed Breakdown**

| Seed | A\_fail Peak | A\_rand Peak |
| :---: | :---------: | :---------: |
| 41 | 76.29% | 11.34% |
| 42 | 74.23% | 11.34% |
| 43 | 81.44%\* | 10.31% |
| 44 | 79.38% | 11.34% |
| 45 | 77.32% | 11.34% |

\*Seed 43 is the best-performing seed, but **not** the primary reported number.

### 4.3 Interpretation

1. **Effect size:** +66.60pp is a large effect with **5/5 seeds** showing A\_fail > A\_rand (same direction as the 1.5B G3 paired test in Appendix C.1)
2. **Mechanism:** Failure-directed selection concentrates learning signal on error regions
3. **Capacity vs strategy:** A\_rand@7B ≈ 11% shows that capacity alone is insufficient; **which examples are curated** matters more than undifferentiated extra volume
4. **Reproducibility:** Low SD on A\_fail peak (2.79pp) indicates a robust in-domain effect across seeds

### 4.4 Mechanism Validation (1.5B Supplementary)

To confirm that the 7B effect reflects the same mechanism observed at smaller scale, we conducted parallel experiments on Qwen2.5-1.5B (see Appendix C for full tables):

* **G3:** A\_fail vs A\_rand replicates the pattern (Δ = +13.0±4.5pp, 5/5 seeds)
* **M1:** Leave-one-type-out analysis shows no prompt variant is necessary for gains
* **M2:** Dose-response curve shows that **selection strategy** dominates over **data quantity** (Δ consistent across K=0.5×, 1.0×, 2.0×)

**Conclusion:** The 7B result is a capacity-amplified version of the same data-centric mechanism.

***

## 5\. Negative Controls: What Does NOT Work \(RQ2\)

### 5.1 M3: Hard Rollback

**Hypothesis:** Locking to early peak checkpoint prevents catastrophic forgetting and preserves gains.

**Result:**

* Peak accuracy: **−6.19pp** relative to non-rollback baseline
* Behavior: Locks to Round 2, prevents later exploration
* **Interpretation:** Premature commitment to early performance is harmful

### 5.2 P2a: Naive Scaling

**Hypothesis:** Adding more data from the same pool improves absolute performance.

**Result:**

* Early peak typically at **round 2**, then **monotonic decline** through later rounds (final accuracy drops sharply; large drawdown)
* **Interpretation:** Distribution-shifted extras (e.g., much longer product SMILES) degrade later-round performance when naively merged—adding volume without distribution control is not curation

### 5.3 P1: LR Decay Adjustment

**Hypothesis:** Adjusting learning rate decay (0.85 → 0.90) can stabilize seed 45's high drawdown.

**Result:**

* Peak accuracy dropped from 77.32% to 70.10%
* Failed pre-registered stability gate (−2pp threshold)
* **Interpretation:** Local hyperparameter tweaks cannot override default recipe without risk

### 5.4 Summary of Negative Controls

**Table 3: Negative Control Summary**

| Experiment | Intervention | Result | Implication |
| :--------- | :----------- | :----- | :---------- |
| M3 | Hard rollback to early peak | Peak −6.19pp | Premature commitment harmful |
| P2a | Naive same-pool scaling | Early peak, then collapse | Wrong distribution ≠ curation |
| P1 | LR decay tweak | Peak −7.22pp | Default recipe is robust |

**Theoretical implication:** These negative results constrain the space of "what counts as bounded improvement." Not all closed-loop interventions qualify; specifically, interventions that ignore distribution shift or commit prematurely fail.

***

## 6\. Generalization Probe: Zero Transfer \(RQ3\)

### 6.1 Hold-Out Set Construction

**Goal:** Test whether in-domain gains transfer to genuinely unseen reactions.

**Construction criteria:**

1. Zero ID overlap with train∪eval
2. **Zero full reaction-key overlap** with train∪eval (stricter than ID check)
3. Valid SMILES format

**Result:** Clean external set with **n=31** samples

### 6.2 Type Overlap Analysis

**Important caveat:** While reaction keys are disjoint, we must distinguish:

* **Key-level disjointness:** No (reactants → product) tuple appears in both sets (verified)
* **Type-name overlap:** 14 of 18 type labels in the clean set also appear in train∪eval (with different instances)

**Table 4: Clean External Set by Reaction Type**

| Reaction Type | Samples | Type Name in Train∪Eval? |
| :------------ | :-----: | :----------------------: |
| Friedel-Crafts | 4 | Yes |
| coupling | 4 | **No** |
| hydrolysis | 3 | Yes |
| rearrangement | 3 | **No** |
| Grignard | 2 | Yes |
| oxidation | 2 | Yes |
| Mannich | 2 | Yes |
| Wolff-Kishner | 1 | **No** |
| Claisen | 1 | Yes |
| nitration | 1 | **No** |
| halogenation | 1 | Yes |
| amidation | 1 | Yes |
| sulfonation | 1 | Yes |
| Wittig | 1 | Yes |
| Suzuki | 1 | Yes |
| aldol | 1 | Yes |
| esterification | 1 | Yes |
| cyclization | 1 | Yes |

**Summary:** 14 types have the same name but different instances; 4 types (coupling, rearrangement, Wolff-Kishner, nitration) are entirely new.

### 6.3 Generalization Results

**Table 5: Clean Hold-Out Evaluation (n=31)**

| Model | Accuracy | Interpretation |
| :---- | :------: | :------------- |
| Base (zero-shot) | **0.00%** | No task capability |
| LoRA seed43 | **0.00%** | In-domain effective, out-of-domain zero |

**Per-type breakdown:** All 18 types achieved 0% accuracy.

### 6.4 Why Skip Multi-Seed External Evaluation?

Since seed43 LoRA (the best-performing in-split seed) achieved 0% on the clean external set, testing additional seeds under the same protocol provides diminishing marginal information. The null result is strong enough to support the conclusion: **cross-split generalization is not observed**.

### 6.5 Interpretation

1. **In-domain gains do not transfer:** The 77.73% in-split accuracy reflects learning of training-distribution patterns, not generalizable chemical rules
2. **Type-name overlap is insufficient:** Even when the model has seen the same type name (e.g., "Friedel-Crafts"), it cannot predict products for unseen instances
3. **Exact-match is strict:** 0% does not mean "completely wrong in all ways" — partial structure matches may exist, but exact SMILES match is the task metric

**Conclusion for RQ3:** Cross-split generalization under exact-match evaluation is not observed in this setting.

***

## 7\. Discussion

### 7.1 Returning to the Three Layers and RQ0

**Epistemic motif.** The empirical pattern matches a simple but consequential claim: in both research and everyday problem solving, **lessons from failure can matter more than lessons from success**. A\_fail operationalizes that claim as a training policy; A\_rand (and naive scaling) largely rehearse undifferentiated experience and yield little under exact-match few-shot chemistry.

**Layer I (RSI-level):** Recursive updates *can* produce auditable gains, but only under protocol fidelity; drawdown control via hard rollback failed; cross-split transfer failed—marking clear **boundaries**.

**Layer II (mechanism):** The operative lever is **failure-directed data curation**, not unconstrained scheduling or naive scaling.

**Layer III / RQ0:** Failure-directed curation yields large, reproducible **in-domain** gains (+66.60pp). Gains are **bounded**: they do not transfer to key-disjoint hold-out data under exact SMILES match, and thus do not establish open-ended RSI. Learning from failure is powerful *inside* a distribution; it is not automatically a passport to open-ended self-improvement.

### 7.2 Relation to RSI

**Table 6: Bounded Refinement vs RSI**

| Criterion | RSI Claim | This Work |
| :-------- | :-------: | :-------: |
| Reproducible improvement under fixed protocol | Required | Demonstrated (in-domain) |
| Generalization to unseen data | Required | Not observed |
| Multi-task / multi-domain validation | Required | Single task only |
| No evaluation contamination | Required | Audited (with retraction) |

**Positioning:** This work demonstrates **bounded data-centric refinement**, not open-ended RSI. The negative hold-out result is an important boundary condition.

### 7.3 Limitations

1. **Data scale:** 46 training samples, 23 reaction types — a **relatively small / few-shot** regime, not large-scale supervision
2. **Hold-out scale:** n=31 with 18 type labels — small sample, wide confidence intervals
3. **Single task:** Reaction product prediction only; no cross-task validation
4. **LoRA, not full fine-tuning:** May limit adaptation capacity
5. **Multi-seed external evaluation:** Skipped due to strong null result
6. **Type-name overlap:** 14/18 types in hold-out have same names as training, complicating "unseen type" claims

### 7.4 Lessons on Evaluation Integrity

The contaminated external set incident (§3.4) highlights several methodological lessons:

1. **ID-level deduplication is insufficient:** Full reaction-key audit is necessary
2. **Contamination produces false positives:** 63% on contaminated set vs 0% on clean set
3. **Retraction is necessary:** Scientific integrity requires correcting the record
4. **Pre-registration helps:** Audit standards should be set before evaluation

### 7.5 Implications for Chemical LLM Evaluation

We recommend that future work on closed-loop chemical LLM improvement adopt:

1. **Key-level leakage audit** (not just ID deduplication)
2. **Pre-registered evaluation protocols**
3. **Transparent retraction policies**
4. **Bounded claims** (in-domain vs generalization)

***

## 8\. Conclusion

We have studied failure-directed data curation as a mechanism for bounded refinement in few-shot reaction product prediction. Our findings:

1. **RQ1 (In-domain effect):** Failure-directed augmentation outperforms random augmentation by a large margin (+66.60pp, N=5, 5/5 seeds), indicating that **failure-localized selection** dominates undifferentiated extra volume under matched nominal budgets.
2. **RQ2 (Negative controls):** Hard rollback, naive scaling, and LR decay tweaks all fail to improve (and can harm) performance, constraining which closed-loop interventions count as improvement.
3. **RQ3 (Generalization):** On a clean hold-out set with zero reaction-key overlap (n=31), both base model and fine-tuned LoRA achieve 0% exact-match accuracy: in-domain gains **are not observed to transfer** under this protocol.
4. **RQ4 (Integrity):** Evaluation contamination can produce false positives for generalization claims; key-level audit and retraction policies are essential.

### 8.1 Bounded Self-Refinement as an Intermediate Claim Class

**Overall conclusion:** Within an RSI research agenda, we instantiate a three-layer path from domain-level questions about verifiable recursive improvement, through a data-centric chemistry loop, to falsifiable RQs. Failure-directed curation yields large, reproducible **in-domain** gains that remain **bounded**—neither cross-split generalization nor open-ended RSI. We propose that **bounded data-centric self-refinement** is a rigorous **intermediate claim class** on the path toward stronger RSI hypotheses. By "intermediate," we mean four jointly necessary properties:

1. **Scope restriction (what may be claimed).** Unlike open RSI—which asserts unbounded capability growth across tasks and distributions—bounded self-refinement restricts claims to (a) a fixed task, (b) a fixed training/eval distribution (here: v9 in-split), and (c) a fixed evaluation protocol (exact SMILES; locked recipe). This restriction is a methodological precondition: it isolates the mechanism (failure-directed curation) from confounds (protocol drift, evaluation contamination) that inflate stronger claims.
2. **Auditable units of progress (what must be measured).** Under a locked recipe, each run emits a pre-specified metric surface—peak, final, and drawdown, plus seed-aggregated mean±SD—so that third parties can re-run and verify numbers. We do **not** claim that accuracy increases monotonically every round (multi-round trajectories can oscillate; hard rollback that freezes an early peak is harmful). Auditability concerns **protocol-defined, reproducible contrasts** (e.g., A\_fail vs A\_rand), not an assumption of unbroken round-wise ascent. Open RSI rhetoric often lacks this property by equating "improvement" with vague "capability growth" without a fixed evaluation surface.
3. **Explicit negative boundaries (what must be denied).** A rigorous intermediate claim class must state *where it fails* with equal force. Our hold-out zero result (§6) and negative controls (§5) delineate failure regions: in-domain gains do not automatically transfer to key-disjoint data; not every closed-loop intervention qualifies as improvement (hard rollback; naive merging of shifted data). These boundaries calibrate expectations for future RSI work as much as the +66.60pp effect does.
4. **Stepping stone, not terminus (what remains open).** Bounded self-refinement is not the final form of RSI. Stronger hypotheses would still need, under the same audit discipline: (a) cross-task transfer (e.g., reaction prediction → retrosynthesis), (b) robustness under distribution shift / novel classes on **clean** hold-outs, and (c) **meta-level** improvement (the curation *policy* itself improving across rounds). In this paper the failure→augment rule remains **human-designed**; we do not claim improver self-modification. Our contribution is a methodological template—fixed protocol, auditable contrasts, explicit negatives—that stronger claims must still satisfy.

**In summary:** We claim neither "RSI achieved" nor "RSI impossible," but a disciplined, falsifiable claim class between naive self-improvement assertions and open-ended RSI. Future work may widen task scope or add meta-learning **while keeping** key-level audits and pre-registered metrics—so that improvement claims stay verifiable rather than aspirational.

**Future work:** Stronger cross-split / cross-type performance will likely need both (i) **larger backbone models** evaluated under locked protocols (not uncontrolled score-chasing), and (ii) **more high-quality, distribution-controlled data** with key-level leakage audits—extending failure-directed curation rather than repeating P2a-style uncritical scaling. Explicit domain adaptation or meta-learning may also help; any stronger claim must still be validated on truly independent hold-outs with pre-registered protocols.

***

## Appendix

### A. Full Hyperparameters and Recipe Hash

**7B Configuration (Primary):**

| Component | Value |
| :-------- | :---- |
| Base model | Qwen2.5-7B [2] |
| Precision | bfloat16 |
| LoRA rank | r=16, α=32 [3] |
| LoRA dropout | 0.05 |
| Target modules | q\_proj, v\_proj, k\_proj, o\_proj, gate\_proj, up\_proj, down\_proj |
| Learning rate | 5e-5, per-round decay 0.85^(r-2), CosineAnnealing (η\_min/η\_max = 0.1) |
| Batch size | 1 |
| Training rounds | 7 (3 epochs/round) |
| Optimizer | AdamW |
| Max sequence length | 256 |
| **Recipe hash** | `9e3788dd7a2eb89d66a1b403d5c5b7c425d583adf7c799c35b761b57fbf7f809` |

**1.5B Configuration (Mechanism Validation):**

| Component | Value |
| :-------- | :---- |
| Base model | Qwen2.5-1.5B [2] |
| Precision | float32 |
| Learning rate | 5e-6, per-round decay 0.85^(r-2) |
| Batch size | 2 |
| Training rounds | 7 |
| **Recipe hash** | `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3` |

### B. Leakage Audit and Contaminated External Set Retraction

**Audit procedure:**

1. Extract (reactant\_keys, product\_key) tuples from each split
2. Compute intersection between train∪eval and external set
3. Full reaction key = canonical SMILES of (reactants, reagents, conditions) → product

**Contaminated set (n=100, deprecated):**

* ID overlap: 0 (pass)
* Full reaction key overlap with eval: **82/100** (fail)
* Reported accuracy: 63% → **RETRACTED**

**Clean set (n=31, current):**

* ID overlap: 0 (pass)
* Full reaction key overlap: **0/31** (pass)
* 18 reaction type labels (14 shared name, 4 entirely new)
* Reported accuracy: 0% / 0%

### C. 1.5B Mechanism Validation (Qwen2.5-1.5B)

To confirm the 7B main effect reflects the same data-centric mechanism at smaller scale, we conducted parallel experiments on Qwen2.5-1.5B under the V5 canonical recipe.

#### C.1 G3: A\_fail vs A\_rand (N=5 seeds)

**Table C1: 1.5B G3 Results — Peak Accuracy**

| Seed | A\_fail Peak | A\_fail Round | A\_rand Peak | A\_rand Round | Δ |
| :---: | :---------: | :----------: | :---------: | :----------: | :---: |
| 41 | 19.59% | R4 | 3.09% | R2 | **+16.50pp** |
| 42 | 14.43% | R2 | 6.19% | R2 | **+8.24pp** |
| 43 | 23.71% | R2 | 5.15% | R2 | **+18.56pp** |
| 44 | 14.43% | R5 | 7.22% | R3 | **+7.21pp** |
| 45 | 16.49% | R2 | 2.06% | R2 | **+14.43pp** |
| **Mean±SD** | **17.73±3.58%** |  | **4.74±1.93%** |  | **+12.99±4.48pp** |

**Table C2: 1.5B G3 Summary**

| Metric | A\_fail | A\_rand | Δ |
| :----- | :----: | :----: | :---: |
| Peak Accuracy | 17.73±3.58% | 4.74±1.93% | **+13.0±4.5pp** |
| Final Accuracy | 10.93±5.35% | 2.47±0.83% | +8.5pp |
| Drawdown | 12.58±2.57% | 3.50±2.17% | +9.1pp |
| Sign consistency | A\_fail > A\_rand in **5/5 seeds** (100%) |
| Paired one-sided *t*-test (exploratory; N=5) | *p* < 0.01 |
| 95% CI for Δ | [8.7, 17.3] pp |

#### C.2 M1: Leave-One-Type-Out Analysis (N=3 seeds)

**Table C3: M1 LOO Results — Paired by Seed**

| Seed | Condition | Peak | Δ vs Full5 | Final | Drawdown |
| :---: | :-------- | :---: | :--------: | :---: | :------: |
| 41 | Full5 baseline | **22.68%** | — | 12.37% | 16.49% |
| 41 | −condition | 15.46% | −7.22pp | 13.40% | 13.40% |
| 41 | −hint | 35.05% | +12.37pp | 14.43% | 20.62% |
| 41 | −format | 21.65% | −1.03pp | 7.22% | 14.43% |
| 42 | Full5 baseline | **19.59%** | — | 11.34% | 11.34% |
| 42 | −condition | 32.99% | +13.40pp | 27.84% | 5.15% |
| 42 | −hint | 19.59% | +0.00pp | 11.34% | 11.34% |
| 42 | −format | 21.65% | +2.06pp | 5.15% | 16.49% |
| 43 | Full5 baseline | **15.46%** | — | 15.46% | 2.06% |
| 43 | −condition | 17.53% | +2.06pp | 11.34% | 7.22% |
| 43 | −hint | 29.90% | +14.43pp | 26.80% | 3.09% |
| 43 | −format | 27.84% | +12.37pp | 26.80% | 1.03% |

**Table C4: M1 Necessity Summary**

| Drop Type | Mean Δ vs Full5 | Seeds ≤ −3pp | Verdict |
| :-------- | :-------------: | :----------: | :------ |
| condition\_variant | +2.75 ± 8.43pp | 1/3 | **Inconclusive** (direction inconsistent) |
| hint\_variant | +8.93 ± 6.37pp | 0/3 | **Not necessary** (removal improves) |
| format\_variant | +4.47 ± 5.73pp | 0/3 | **Not necessary** (removal improves) |

**Note:** synonym\_variant and context\_variant were never invoked by the canonical pipeline (`num_variants` slicing); Δ=0 for those drops is a pipeline fact, not a scientific finding. Full5 baselines here are **within-runner** controls for the M1 script and are **not** numerically identical to Phase3/G3 seed peaks; LOO deltas should be read as paired contrasts against that runner's Full5, not as absolute G3 peaks.

**M1 conclusion:** No prompt variant type satisfies the necessity criterion (Δ≤−3pp with ≥2/3 seed agreement) under the current protocol slice—consistent with gains being driven more by *which failures are selected* than by *which prompt wording* is used.

#### C.3 M2: Dose-Response Curve (K=0.5×/1.0×/2.0×, N=3 seeds)

**Table C5: M2 Raw Results — Peak Accuracy by K**

| Seed | K | A\_fail Peak | A\_rand Peak | Δ |
| :---: | :---: | :---------: | :---------: | :---: |
| 41 | 0.5× | 19.59% | 6.19% | **+13.40pp** |
| 41 | 1.0× | 21.65% | 4.12% | **+17.53pp** |
| 41 | 2.0× | 32.99% | 5.15% | **+27.84pp** |
| 42 | 0.5× | 19.59% | 2.06% | **+17.53pp** |
| 42 | 1.0× | 15.46% | 5.15% | **+10.31pp** |
| 42 | 2.0× | 14.43% | 7.22% | **+7.22pp** |
| 43 | 0.5× | 34.02% | 4.12% | **+29.90pp** |
| 43 | 1.0× | 19.59% | 6.19% | **+13.40pp** |
| 43 | 2.0× | 18.56% | 5.15% | **+13.40pp** |

**Table C6: M2 Aggregated — Mean Δ(K)**

| K | Mean Δ | SD | Min | Max |
| :---: | :----: | :---: | :---: | :---: |
| 0.5× | **+20.27pp** | 7.01pp | +13.40 | +29.90 |
| 1.0× | **+13.75pp** | 2.96pp | +10.31 | +17.53 |
| 2.0× | **+16.15pp** | 8.64pp | +7.22 | +27.84 |

**Caveat:** Per-round augmentation counts were only *nominally* K-matched; A\_rand often accumulated slightly more examples than A\_fail after round 1. Because A\_fail still dominates under that imbalance, the selection effect is unlikely to be an artifact of A\_fail receiving *more* data.

**M2 conclusion:** Δ(A\_fail − A\_rand) is large and consistent across K∈{0.5,1,2}× (including the smallest K), with no reliable monotone dose benefit for A\_rand. This supports **selection strategy** (directed vs random) over **undifferentiated volume** as the main driver.

### D. Clean External Set: Full Sample and Type Table

All 31 samples (with full SMILES, reaction types, and audit metadata) are provided in the Supplementary Material.

### E. Negative Result Trajectories

**Figure E1: P2a Naive Scaling — Catastrophic Forgetting Pattern**

![Figure E1. P2a naive scaling trajectory](figures/figE1_p2a_scaling.pdf)

*Schematic trajectory of a representative seed under P2a naive scaling (per-round values: 15 → 22 → 18 → 14 → 10 → 7 → 4%). The early peak at R2 is followed by monotonic decline (catastrophic forgetting). Per-seed trajectories and peak/final aggregates are reported in §5.2 and Tables 2–3; the plotted values illustrate the qualitative pattern and are not extracted from any single seed's log.*

**Figure E2: M3 Hard Rollback — Premature Lock**

![Figure E2. M3 hard-rollback trajectory](figures/figE2_m3_rollback.pdf)

*Schematic trajectory of a representative seed under M3 hard rollback (per-round values: 12 → 18 → 18 → 18 → 18 → 18 → 18%). The rollback locks the checkpoint at the R2 peak (dashed line), preventing further exploration; a no-rollback baseline continues to improve (see main text §5.1). Peak gap: −6.19pp relative to the no-rollback baseline reported in §5.1. The plotted values illustrate the qualitative pattern and are not extracted from any single seed's log.*

***

## Acknowledgments

We thank Zhejiang Lab for providing the Zero2X computational infrastructure that supported the experiments in this work. We also thank the 021SFM scientific corpus team for their contributions to data preparation and theoretical discussion during the early stages of this project.

***

## Funding

This work was supported by Zhejiang Lab.

***

## Competing Interests

The authors declare no competing interests.

***

## Author Contributions

**Dong Li:** Conceptualization, Writing—original draft. **Yixiong Wei:** Methodology, Investigation. **Shi Luo:** Software, Investigation. **Yingjun Yuan:** Software, Investigation.

***

## Ethics Statement

This work uses publicly available chemical reaction datasets and does not involve human subjects, animal experimentation, or sensitive data. All experimental data are derived from standard chemical databases and internal curated corpora used under institutional data-sharing agreements. No ethical approval was required.

***

## Data and Code Availability

The clean external evaluation set (n=31, with full SMILES, reaction-type labels, and reaction-key audit metadata), the v9\_g1prime train/eval split, recipe hashes, evaluation and leakage-audit scripts, and the full replication package (training, augmentation, negative-control, and 1.5B mechanism-validation runners) are publicly archived at **[REPO\_URL]** (Zenodo DOI: **[DOI]**). The Mermaid source and rendering script for Figure 1, and the matplotlib source for Appendix Figures E1–E2, are included under `figures/`. The base model (Qwen2.5-7B) is publicly available on Hugging Face; trained LoRA adapters can be provided upon request.

***

*Recipe hash (7B):* `9e3788dd7a2eb89d66a1b403d5c5b7c425d583adf7c799c35b761b57fbf7f809`
*Recipe hash (1.5B):* `2a965c392fee090d794a3857a03a69e36dbf16276e6d534a7b78c77a60391dc3`

## References

[1] J. Schmidhuber. "Evolutionary Principles in Self-Referential Learning." Diploma thesis, Technische Universität München, 1987. (Early self-referential / self-improvement framing)

[2] Qwen Team. "Qwen2.5 Technical Report." *arXiv:2412.15115*, 2024.

[3] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.

[4] D. Weininger. "SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules." *Journal of Chemical Information and Computer Sciences*, 28(1):31–36, 1988.

[5] B. Settles. "Active Learning Literature Survey." Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009.

[6] Y. Bengio, J. Louradour, R. Collobert, J. Weston. "Curriculum Learning." *ICML*, 2009.

[7] D. Zha, Z. P. Bhat, K.-H. Lai, F. Yang, Z. Jiang, S. Zhong, X. Hu. "Data-centric Artificial Intelligence: A Survey." *arXiv:2303.10158*, 2023.

[8] L. Zhang et al. "A Survey on Data Selection for LLM Instruction Tuning." *arXiv:2402.10980*, 2024.

[9] J. Huang, S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, J. Han. "Large Language Models Can Self-Improve." *EMNLP*, 2023.

[10] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, P. J. Liu. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*, 21(140):1–67, 2020.

[11] R. Gómez-Bombarelli et al. "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules." *ACS Central Science*, 4(2):268–276, 2018.

[12] K. Yang et al. "Analyzing Learned Molecular Representations for Property Prediction." *Journal of Chemical Information and Modeling*, 59(8):3370–3388, 2019.

[13] P. Schwaller, T. Laino, T. Gaudin, P. Bolgar, C. A. Hunter, C. Bekas, A. A. Lee. "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction." *ACS Central Science*, 5(9):1572–1583, 2019.

[14] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl. "Neural Message Passing for Quantum Chemistry." *ICML*, 2017.

[15] C. W. Coley, W. H. Green, K. F. Jensen. "Machine Learning in Computer-Aided Synthesis Planning." *Accounts of Chemical Research*, 51(5):1281–1289, 2018.

[16] D. Zhang, W. Liu, Q. Tan, J. Chen, H. Yan et al. "ChemLLM: A Chemical Large Language Model." *arXiv:2402.06852*, 2024. (ChemLLM / ChemData / ChemBench framework)

[17] I. V. Tetko, P. Karpov, R. Van Deursen, G. Godin. "State-of-the-art augmented NLP transformer models for direct and single-step retrosynthesis." *Nature Communications*, 11:5575, 2020. (USPTO-50K top-5 ≈ 84.8%)

[18] Z. Tu, C. W. Coley. "Permutation Invariant Graph-to-Sequence Model for Template-Free Retrosynthesis and Reaction Prediction." *Journal of Chemical Information and Modeling*, 62(16):3858–3871, 2022. (Graph2SMILES)

[19] Z. Zhao, D. Ma, L. Chen, L. Sun, Z. Li et al. "Developing ChemDFM as a large language foundation model for chemistry." *Cell Reports Physical Science*, 6(1):102523, 2025.

[20] Z. Zhong, J. Song, Z. Feng, T. Liu, L. Jia et al. "Recent advances in deep learning for retrosynthesis." *WIREs Computational Molecular Science*, 13(5):e1694, 2023.