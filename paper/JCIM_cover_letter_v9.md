# Cover Letter — Journal of Chemical Information and Modeling

**Manuscript title:** Failure-Directed Reaction-Data Curation for Few-Shot Exact-SMILES Product Prediction: In-Domain Gains under a Chemical LLM Evaluation Protocol without Cross-Split Generalization

**Authors:** Dong Li, Yixiong Wei, Shi Luo, Yingjun Yuan  
**Corresponding author:** Dong Li (lidong@zhejianglab.org)  
**Affiliation:** Center for Earth and Space Science Computing, Zhejiang Lab, Hangzhou, China

---

Dear Editors,

We submit the enclosed manuscript for consideration as an Article in *Journal of Chemical Information and Modeling*.

**Fit to JCIM.** The work addresses machine learning on chemical reaction data with two coupled contributions suited to chemical information and modeling: (1) a **failure-directed reaction-data curation** policy for exact canonical SMILES product prediction under a scarce-label budget; and (2) a reusable **chemical LLM evaluation protocol**—recipe locking, reaction-key (not merely ID) leakage audit, pre-registered in-split versus hold-out gates, and transparent retraction of a contaminated external score. The primary empirical claim is curation efficacy under protocol fidelity, together with an explicit negative boundary (0% on a key-disjoint hold-out).

**Main results (locked Qwen2.5-7B + LoRA, N=5 seeds).** Failure-directed augmentation reaches 77.73%±2.79% peak in-split exact-match accuracy versus 11.13%±0.46% for budget-matched random augmentation (Δ = +66.60pp). Hard rollback, naive merging of distribution-shifted extras, and a local learning-rate tweak all fail as improvement interventions. On a clean hold-out with zero reaction-key overlap with train∪eval (n=31), both the base model and the fine-tuned LoRA achieve 0% exact-match accuracy.

**Baselines and why USPTO multi-shot tables are not the primary comparison.** We respectfully ask reviewers to judge the manuscript against its stated scientific question—*under a fixed few-shot budget and exact SMILES scoring, does failure-directed reaction-data curation outperform budget-matched random curation on a locked LLM+LoRA recipe?* Strong USPTO-50K results for Molecular Transformer–class and Graph2SMILES models are important **capability references for data-rich, multi-shot** regimes (tens of thousands of paired examples). They are **not** fair head-to-head competitors for our protocol: retraining those architectures on 46 labeled reactions would change the contrast into an architecture/data-regime bake-off and would still not isolate the curation policy that is our object of study. Accordingly, the primary baseline is **A_rand** under matched nominal augmentation budget; USPTO and chemical-LLM literature contextualize the setting in Related Work (§2.1) rather than serving as a missing leaderboard table.

**Data and code.** The replication package (splits, hold-out with audit metadata, recipe hashes, runners, and leakage-audit scripts) is publicly available at https://github.com/ldd19990713-coder/rsi-failure-directed-curation (Zenodo DOI: 10.5281/zenodo.24391973).

We confirm that the manuscript is original, not under consideration elsewhere, and that all authors approve submission. We suggest the manuscript be considered under JCIM topics spanning machine learning on chemical data, reaction information, and evaluation methodology for chemical predictors.

Thank you for your consideration.

Sincerely,  
Dong Li  
On behalf of all co-authors  
Center for Earth and Space Science Computing, Zhejiang Lab  
lidong@zhejianglab.org

---

## 中文要点（供作者自用，勿随稿提交）

- 投稿定位：反应数据策展 + 化学 LLM 评测协议，非 RSI 论文、非 USPTO 排行榜文。
- Cover letter 必须写清：主对照是 A_fail vs 等预算 A_rand；USPTO 多样本结果不可直接当缺 baseline。
- 建议走 ACS 订阅通道（无强制 APC）；可选 OA 另议。
