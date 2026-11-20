# 论文撰写提纲（科学终稿版）

**依据**: 桌面 `RSI_final_outputs_v3.tar.gz` + 既有机制链裁决  
**评判结论**: 主证据链足以支撑一篇 **有界、数据中心、协议严格** 的实证论文；须以 **科学问题** 驱动，把 in-split 增益与 **跨分布零泛化** 写成同一理论故事的两面，禁止 RSI/生产/脏外集 63% 叙事。

**建议题目（英）**  
Failure-Directed Data Curation for Few-Shot Reaction Product Prediction:  
In-Domain Gains under Protocol Fidelity without Cross-Split Generalization

**建议题目（中）**  
少样本反应产物预测中的失败驱动数据策展：协议保真下的域内增益与跨划分零泛化

**建议投稿定位**: 实证/方法短文或 workshop 全文（强调协议、消融与负结果，而非 SOTA 刷榜）

---

## 0. 综合评判（写提纲前的判定）

### 0.1 站得住的主张

| 科学命题 | 证据 | 强度 |
| :--- | :--- | :---: |
| 在无泄露 few-shot 设定下，**失败样本定向增广**显著优于等设定随机增广 | 7B+LoRA，N=5：77.73%±2.79% vs 11.13%±0.46%（**+66.60pp**） | 高 |
| **协议保真**是因果结论前提 | 早期漂移伪阴性、fp16 NaN、脏外集 63% 假阳性 | 高 |
| 硬回滚 / 朴素并入分布偏移数据 **不能**替代策展 | M3 冒烟 peak−6.19pp；P2a 早峰后崩 | 中高 |
| 域内成功 **不蕴含** 跨划分泛化 | 干净外集 n=31，基座与 LoRA seed43 均为 **0%**（审计重叠=0） | 高（对「无泛化」这一负命题） |

### 0.2 必须改准的表述（相对包内 outline v3）

| 包内写法风险 | 建议改法 |
| :--- | :--- |
| 「18 种**全新**反应类型」 | 改为：与 train∪eval **反应键零重叠** 的 hold-out 反应，覆盖 **18 个类型标签**；其中若干标签名或在训练集出现过，但 **实例与键均未见于训练/评测划分**。0% 支持「跨划分/跨实例失效」，勿过度宣称「一切类型标签皆训练未见」除非另附类型交集表。 |
| M3「略低于基线」 | 写清：硬回滚锁 early peak，相对无回滚 **peak −6.19pp**（失败线） |
| P2a「无显著提升」 | 写清：并入长 SMILES/偏移样本后 **早峰后灾难性遗忘**（final 大跌） |
| 「策展质量 >> 数据量」单独成立 | 与 M2 剂量曲线、P2a 共同论证：**定向选样**为主；盲目加量/错分布有害 |
| 主数字 81.44% | 正文主数字仅 **77.73%±2.79%** |

### 0.3 理论与实验如何咬合（全文主线）

**理论命题**: 在表示能力足够但监督极稀缺时，闭环改进的信息瓶颈在 **错误驱动的数据修复（data-centric）**，而非无约束的递归权重调度或同池堆量；且此类改进默认是 **训练分布上的有界精炼**，不能先验外推到键级 hold-out。

**实验操作化**:  
(1) 固定协议与泄露门禁 → (2) A_fail vs A_rand → (3) 调度/扩容负对照 → (4) 容量（7B+LoRA）放大效应 → (5) 干净 hold-out 检验外推。

---

## 1. 科学问题（Introduction 开篇即抛出）

**总问题 (RQ0)**  
在严格无泄露的少样本化学反应产物预测（精确 SMILES）中，何种闭环干预能带来可复现的有界改进，其边界在哪里？

**分问题**

| ID | 问题 | 对应实验 |
| :---: | :--- | :--- |
| **RQ1** | 失败驱动增广是否优于预算匹配的随机增广？ | A_fail vs A_rand（7B+LoRA，N=5）；辅以 1.5B 机制链 |
| **RQ2** | 递归调度类干预（硬回滚）与同池朴素扩容能否进一步提升？ | M3；P2a/P2a-S1 |
| **RQ3** | 域内增益能否转移到与 train∪eval 反应键不相交的 hold-out 集？ | 干净外集 n=31，基座 vs LoRA |
| **RQ4** | 协议漂移与评测污染如何导致假阴性/假阳性？ | 漂移考古；脏外集 82/100 重叠更正 |

全文每一节用 RQ 编号回扣，避免写成实验流水账。

---

## 2. 摘要（重点撰写 · 中英）

### 2.1 中文摘要（建议稿）

少样本化学反应产物预测在精确 SMILES 匹配下极难，且闭环「自改进」主张易被协议漂移与评测污染夸大或掩盖。本文在泄露受控划分（训练 46 / 评测 97）上，系统检验 **失败驱动数据策展** 作为有界自精炼机制的有效性与边界。理论动机是：稀缺监督下，可验证增益应优先来自对模型错误的定向数据修复，而非无约束的递归权重操作或忽视分布偏移的堆量。实验上，我们冻结 Qwen2.5-7B + LoRA（bf16）配方，在 N=5 种子下比较失败定向增广（A_fail）与随机增广（A_rand）：A_fail 峰值准确率达 **77.73%±2.79%**，A_rand 仅为 **11.13%±0.46%**（**+66.60pp**）。对照实验表明，轮末硬回滚会锁死早期峰值并损害性能；将分布偏移的额外样本并入训练会诱发早峰后遗忘；局部学习率衰减调整亦未通过稳化门禁。进一步，在与 train∪eval **反应键零重叠** 的干净 hold-out 集（n=31，含 18 类类型标签）上，基座与已训 LoRA 的准确率均为 **0%**。我们还记录并撤回曾被污染外集抬高的虚假泛化数字。结论：失败驱动策展可在同分布内带来大幅、可复现的增益，但该增益是 **域内有界的**；跨划分泛化在本设定下 **未被观测到**。本文因此将贡献定位为协议严格的数据中心有界精炼，而非持续自我改进（RSI）或跨类型通用化学推理的验证。

### 2.2 English abstract（建议稿）

Few-shot reaction product prediction under exact SMILES matching is brittle, and claims of closed-loop “self-improvement” are easily distorted by protocol drift or contaminated evaluation. We study **failure-directed data curation** as a mechanism for *bounded* refinement under a leakage-controlled split (46 train / 97 eval). Theoretically, when supervision is scarce, verifiable gains should arise primarily from repairing model failures via targeted data, rather than unconstrained recursive weight updates or uncritical data scaling. Empirically, with a frozen Qwen2.5-7B + LoRA (bf16) recipe over N=5 seeds, failure-directed augmentation (A_fail) reaches **77.73%±2.79%** peak accuracy versus **11.13%±0.46%** for matched random augmentation (A_rand; **+66.60pp**). Hard rollback harms peak by locking early checkpoints; naively merging distribution-shifted extras induces early-peak collapse; a local LR-decay tweak fails a pre-registered stability gate. On a held-out set with **zero reaction-key overlap** with train∪eval (n=31; 18 type labels), both the base model and the fine-tuned LoRA obtain **0%** exact-match accuracy. We also retract a previously reported external score inflated by eval-key duplication. We conclude that failure-directed curation yields large, reproducible **in-domain** gains that do **not** transfer under our held-out protocol, supporting bounded data-centric refinement rather than open-ended recursive self-improvement or cross-split chemical generalization.

### 2.3 摘要必须出现 / 禁止出现

| 必须 | 禁止 |
| :--- | :--- |
| 77.73%±2.79%，+66.60pp，N=5 | 主打 81.44% |
| 干净 hold-out 0% | 脏外集 63% 作泛化 |
| 有界 / 非 RSI | 「RSI 已验证」「可生产」 |
| 协议/泄露门禁 | 「策略绝对重于容量」这类绝对化 |

---

## 3. 全文结构提纲（理论 ↔ 实验对照）

### §1 Introduction（科学问题先行）

1. 现象：exact SMILES + 少样本使「看起来能涨分」的闭环极不可靠。  
2. 提出 RQ0–RQ4。  
3. 立场：检验 **data-centric bounded refinement**，明确 **非** open RSI。  
4. 贡献列表（与 RQ 一一对应）：协议方法；A_fail 主效应；负对照；hold-out 零泛化；污染更正的科学诚信。

### §2 Related Work（为 RQ 服务，勿堆砌）

| 小节 | 理论钩子 | 后文实验如何回应 |
| :--- | :--- | :--- |
| 2.1 化学序列/SMILES 预测与少样本 | 任务难度与评测严苛性 | 解释高方差与 0% hold-out |
| 2.2 困难样本/失败驱动/主动学习 | 「修错误」vs「加随机数据」 | → §4 A_fail vs A_rand |
| 2.3 数据中心 AI vs 算法中心调参 | 策展优先 | → §5 扩容/回滚失败 |
| 2.4 RSI / self-refinement 文献 | 开放自我改进的强主张 | → 本文有界负边界（§6） |

### §3 Problem Setup & Protocol Fidelity（方法=可检验性理论）

**理论点**: 无门禁的闭环实验无法区分「真增益」与「漂移/泄漏」。  

**内容**:  
- 任务：条件化产物 SMILES，exact match。  
- 划分 v9：46/97；泄露审计标准（ID、产物、完整反应键）。  
- 配方锁：7B+LoRA，bf16（禁 fp16），hash `9e3788…`；与 1.5B 机制配方区分。  
- 主指标：peak / final / drawdown；多种子均值±SD。  
- **诚信小节**: 脏外集 82/100 与 eval 键重叠 → 撤回 63%；干净集重建标准。

### §4 Main Result — RQ1（失败定向 vs 随机）

**理论预期**: 若信息瓶颈在错误区域，则定向增广 ≫ 等预算随机。  

**实验**: Table — A_fail vs A_rand（N=5）；可选附图种子条形图。  
**机制铺垫（可放 4.3 或附录）**: 1.5B 上 G3/M1/M2——定向稳健、措辞类型非必要、剂量曲线显示选样主导——说明 7B 主效应是 **同机制在容量上的放大**，非换题。  
**说法**: 「容量放大定向收益；A_rand@7B≈11% 表明单靠容量不够」。

### §5 Negative Controls — RQ2（理论边界）

| 实验 | 理论假说 | 结果（准确写） | 含义 |
| :--- | :--- | :--- | :--- |
| M3 硬回滚 | 「拒绝变差」可降回撤且保 peak | 冒烟 peak **−6.19pp**，锁 R2 | 过早拒绝探索有害 |
| P2a 朴素扩容 | 更多同池数据抬绝对水平 | 复杂度/分布偏移 → 早峰后遗忘 | 错分布堆量 ≠ 策展 |
| P1 LR 衰减 | 可压 seed45 DD | peak 掉出 −2pp 门禁 | 不改默认配方 |

本节强调：**负结果也是对「何种闭环算改进」的理论约束**。

### §6 Generalization Probe — RQ3

**理论预期**: 若学到的是可迁移化学规则，键级 hold-out 应保留部分准确率；若仅为域内记忆/窄模式，exact match 可崩至 0。  

**实验**:  
- 构建 `smiles_clean`：n=31，与 train∪eval 反应键重叠 0（附审计）。  
- 基座 0%；LoRA seed43 0%；类型标签分解（附录全表）。  
- 说明为何跳过多种子：单检查点已达零，同协议边际信息低。  

**讨论用词**: 「跨划分泛化未观测到」；避免无交集表时写「全部类型训练中从未出现」。

### §7 Discussion（理论回收）

1. 回答 RQ0：可复现的有界增益来自失败驱动策展；边界是训练分布。  
2. 与 RSI：定义对照表（开放目标 / 保留 / 跨任务 vs 本文）。  
3. 局限：n_train=46；hold-out n=31；单任务；LoRA 非全量；外测多种子未做。  
4. 启示：评测门禁（键级去重）应成为闭环化学 LLM 工作的标准部件。

### §8 Conclusion

三条结论对应 RQ1–RQ3；一句未来工作：类型覆盖扩展或显式域适应（须另设 holdout），而非宣称更大模型即泛化。

### 附录

A 完整超参与 hash  
B 泄露与脏外集更正  
C 1.5B 机制补遗（G3/M1/M2）  
D 干净外集样本与类型全表  
E 负结果轨迹图  

---

## 4. 图表清单（建议）

| 编号 | 内容 | 服务 RQ |
| :---: | :--- | :---: |
| Fig.1 | 方法示意图：失败→增广→多轮 FT→评测门禁 | RQ0 |
| Tab.1 | A_fail vs A_rand，N=5 均值±SD | RQ1 |
| Tab.2 | 负对照汇总（M3/P2a/P1） | RQ2 |
| Tab.3 | 干净 hold-out 审计 + 0%/0% | RQ3 |
| Fig.2 | （可选）P2a 与 M2 逐轮对比：早峰后崩 | RQ2 |
| Fig.3 | （可选）脏外集分层 vs 干净外集：诚信与方法教训 | RQ4 |

---

## 5. 写作原则（准确合理）

1. **先问题后数字**：每节首句回扣 RQ。  
2. **主数字唯一**: 77.73%±2.79%；81.44% 仅脚注「best seed」。  
3. **因果措辞分级**: 「支持 / 表明 / 与…一致」；对 0% 用「未观测到泛化」，慎用「证明模型零化学知识」（基座与 FT 在 exact 指标下的行为）。  
4. **理论一句 ↔ 实验一表** 成对出现。  
5. **负结果与正结果同级标题**，避免藏在附录。  
6. 全文检索禁词：RSI verified、生产就绪、外集 63%、有限泛化已验证。

---

## 6. 推荐撰写顺序

1. 定稿摘要中英 + RQ 表（本文 §1–§2）  
2. §3 协议与泄露更正（诚信）  
3. §4 主表 + §5 负对照  
4. §6 干净外集 0%  
5. §2 相关工作与 §7 讨论  
6. 附录补 1.5B 机制与轨迹  

---

## 7. 一句话总判

**可以按本提纲开写正式初稿**：科学问题是「失败驱动策展能否以及在何种边界上带来可复现有界改进」；摘要须同时给出 **+66.6pp 域内效应** 与 **hold-out 0%**；理论核心是数据中心有界精炼，实验正负结果共同支撑，表述上严守协议与去污染后的口径。
