# Digital Discovery 投稿详细步骤（针对本文 v7）

> 依据 RSC *Author guidelines for Digital Discovery*（2026 起 APC 规则）与本稿现状定制。  
> 官方指南：https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery  
> 期刊主页：https://www.rsc.org/publishing/journals/digital-discovery  
> 本稿英文稿：`paper/RSI_paper_v7.md`  
> 代码/数据：https://github.com/ldd19990713-coder/rsi-failure-directed-curation · DOI `10.5281/zenodo.24391973`

---

## 0. 投稿定位（先定死）

| 决策项 | 建议 | 理由 |
|:---|:---|:---|
| **文章类型** | **Full Paper**（勿投 Communication） | Communication 通常 ≤4 印刷页，需“紧急高新颖性”辩护；本稿含负结果、撤回、附录机制实验，适合 Full Paper（无硬性页限，但长度须与内容匹配） |
| **语言** | 英文稿 `RSI_paper_v7.md` | 期刊仅接受英文 |
| **审稿** | 单盲（作者可知）；可选 Transparent peer review | 投稿时可勾选是否公开审稿意见 |
| **开放获取** | Gold OA；接受后付 APC | 2026 年起 Full price **£2,200**（+税）；印度等有减半；R4L 可全免；**先查之江实验室是否有 RSC OA 协议** |
| **范围匹配话术** | 化学 LLM + 数据中心策展 + 评测诚信/可复现 | 对准 Digital Discovery “data first / ML·AI for chemistry” |

**关键缺口（投前必须处理）：** Digital Discovery 对 **LLM fine-tuning** 明确要求：微调开源权重模型后应 **分享更新后的模型权重**；DAS 中 **“Data available upon request” 一般不可接受**。本稿目前写 LoRA “upon request”——需改为公开（Hugging Face / Zenodo）或提供可复现的适配器归档 DOI。

---

## 1. 投前准备清单（约 3–7 天）

### 1.1 经费与权限

1. 用 [RSC Journal Finder](https://www.rsc.org/publishing/journal-finder) 查 **Zhejiang Lab / 之江实验室** 是否覆盖 Digital Discovery APC。  
2. 确认通讯作者（Dong Li）邮箱、ORCID 与 ScholarOne 账号一致。  
3. 全体作者确认署名顺序、单位、通讯作者（可双通讯，须在稿首页与 cover letter 标明）。  
4. 准备 APC 支付路径（协议减免 / 实验室经费 / RSC 会员 15% 折扣等）。

### 1.2 满足 Digital Discovery 数据/代码硬性政策

| 要求 | 本稿现状 | 行动 |
|:---|:---|:---|
| 审稿人可访问 code + data | GitHub 公开 + Zenodo DOI | 保持公开；DAS 写全 URL（`https://doi.org/...`） |
| 持久归档 + DOI | Zenodo `10.5281/zenodo.24391973` | DAS 同时可写概念 DOI `10.5281/zenodo.24391972`（始终指向最新版） |
| **微调后分享权重** | LoRA 仅 upon request | **上传 seed43（及建议全部 5 seeds）LoRA 到 HF 或 Zenodo，写入 DAS** |
| LLM 推理日志（输入/输出） | 未整理 | 整理代表性日志（至少 hold-out 与若干 eval 样例）放入仓库 `logs/` 或 SI |
| 写明模型标识 | Qwen2.5-7B / 1.5B + recipe hash | 在方法节保持；可补 Hugging Face model card 链接 |
| 训练/测试完整数据 | v9 train/eval + holdout 已公开 | DAS 明确列出三个路径 |

### 1.3 文稿内容微调（对准期刊）

1. **Abstract**：单段、**约 50–250 词**；少写方法细节，突出目标、结果与对化学数字发现/评测诚信的意义。  
2. **Title**：可略偏“化学 LLM / evaluation integrity / data curation”，便于检索（现标题可用，亦可微调）。  
3. **Introduction**：加强与 *Digital Discovery* 读者相关的动机（可复现化学 AI、泄露审计、失败驱动数据策展）。  
4. **Conflicts of interest**：改为期刊标准句：`There are no conflicts to declare.`  
5. **Data availability**：独立成节，放在 **conflicts 之后、acknowledgements 之前**；标题用 **“Data availability”**（不要只用长段落且夹在致谢后——按指南顺序调整）。  
6. **Author contributions**：建议改用 **CRediT** 术语（Conceptualization, Methodology, Software, Investigation, Writing – original draft, Writing – review & editing, Supervision, Funding acquisition 等）。  
7. **References**：改为 RSC **上标数字**体例；作者姓名全列；格式示例：  
   `A. Name, B. Name and C. Name, Journal Title, year, volume, pages, DOI.`  
8. **引用数据集/代码**：在参考文献中增加 Zenodo/GitHub 数据条目（指南鼓励 data citation）。  
9. 附录 D / SI：指向 `data/holdout_clean/holdout_n31.json`（已改）；可另做简短 SI PDF（可选轨迹表、完整超参）。

### 1.4 格式与模板

1. 下载 RSC **Article template**（Word 或 LaTeX/Overleaf；Communication 模板勿用）。  
   - Word / LaTeX：指南页 “Article templates”  
2. 将 `RSI_paper_v7.md` **转入模板**（推荐 Word 便于 ScholarOne；或 LaTeX + **同时上传 PDF**）。  
3. **Figures**：导出为 **TIFF ≥600 dpi**（或 EPS/PDF，录用后会转 TIFF）。Fig.1 / E1 / E2 已有 PDF/PNG/SVG → 再导出 TIFF。  
4. 图宽注意双栏习惯（单栏约 ≤8.3 cm，双栏更宽）；表题在上、图题在下。  
5. 可选：用 RSC **Submission checker tool** 自检章节与合规声明。

### 1.5 投稿配套文件

| 文件 | 内容 |
|:---|:---|
| **Main manuscript** | Word 或 LaTeX+PDF，含图或文末附图 |
| **Cover letter** | 致 Editor；说明为何适合 Digital Discovery；强调有界主张、负结果与数据诚信；**勿**在信中写 preferred reviewers |
| **Preferred reviewers** | 3–5 名（化学信息学 / 分子 ML / LLM for science）；仅填 ScholarOne |
| **Supplementary Information**（可选） | 额外表、日志说明、完整轨迹 |
| **Graphical abstract / TOC figure**（若系统要求） | ≤8 cm × 4 cm，TIFF ≥600 dpi |

---

## 2. Cover letter 要点（可直接扩写）

建议包含：

1. 稿件题目、文章类型：**Full Paper**、期刊名 **Digital Discovery**。  
2. 一句话贡献：failure-directed curation 在少样本精确 SMILES 任务上相对随机增广有大幅域内增益，并在键级不相交 hold-out 上报告零迁移与评测污染撤回。  
3. 为何适合本刊：化学 AI 可复现性、数据/代码开放、评测协议（与 “data first” 一致）。  
4. 确认：无一稿多投；所有作者知情；数据与代码已公开（附 GitHub + Zenodo DOI）；**LoRA 权重已公开（补齐后写上链接）**。  
5. 通讯作者联系方式。

---

## 3. ScholarOne 在线提交步骤

入口：从期刊页 **Submit** → RSC ScholarOne Manuscripts（需弹窗）。  
指南：https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery  

1. **注册/登录** ScholarOne（New User → Register；启用浏览器弹窗）。  
2. **Begin Submission** → 选择期刊 **Digital Discovery** → 文章类型 **Full Paper**（或系统中对应名称）。  
3. 填写题名、摘要、关键词；作者信息与 ORCID。  
4. 上传文件（一次事务内传完；≤5 个文件或 ZIP≤20 文件）：  
   - Manuscript  
   - Figures（若未嵌文）  
   - Cover letter  
   - SI（如有）  
5. 填写 **Data availability** 相关勾选/粘贴 DAS。  
6. 填写 **Conflicts**：无利益冲突。  
7. 填写 **Preferred / non-preferred reviewers**（系统内，不进 cover letter）。  
8. 选择是否 **Transparent peer review**。  
9. 确认金 OA / 许可偏好（CC BY 或 CC BY-NC；以基金要求为准，国内常见 CC BY）。  
10. 预览合并 PDF → **Submit**。  
11. 用 manuscript tracker 跟踪；所有作者应收到确认邮件。

---

## 4. 审稿阶段预期

Digital Discovery 流程概要：

1. Associate Editor 初审。  
2. ≥2 名匿名审稿人 + **可能的 data reviewer**（检查代码能否跑、能否支撑主要结论）。  
3. AE 综合决定（修回 / 拒 / 转投其他 RSC 刊）。  

**应对 data reviewer 的准备：**

- README Quick start 可运行（至少：泄露审计、key_disjoint；训练需 GPU 处写明）。  
- `requirements.txt`、recipe yaml、数据路径正确。  
- LoRA 权重可下载；给出加载 5–10 行示例。  
- 固定种子与 recipe hash 与论文一致。

---

## 5. 修回与录用后

### 5.1 Revision

按编辑信：点对点回复审稿人；高亮修订稿；补充高分辨率图；必要时更新 Zenodo Release（新 DOI 版本）并改 DAS。

### 5.2 Acceptance

1. 确认最终 DAS 含：**最新代码 DOI + 归档版本 DOI**（指南要求）。  
2. 支付 APC（或协议自动覆盖）。  
3. 签署 licence（CC BY / CC BY-NC）。  
4. Proof Central 校对。  
5. （可选）申请 journal cover（另收费）。

---

## 6. 建议时间表（可执行）

| 天数 | 任务 |
|:---:|:---|
| D1 | 查 OA 协议/APC；定 Full Paper；作者确认署名 |
| D1–D2 | **公开 LoRA 权重** + 推理日志样例；更新 README/DAS |
| D2–D4 | MD → RSC Word/LaTeX；图转 TIFF；参考文献改 RSC 体例 |
| D4 | 压缩 Abstract；调整章节顺序（COI → Data availability → Acknowledgements） |
| D5 | Cover letter + preferred reviewers 名单；Submission checker |
| D6 | ScholarOne 上传提交 |
| 之后 | 跟踪审稿；准备 data reviewer 可跑通审计脚本 |

---

## 7. 本稿对照：已就绪 / 未就绪

### 已就绪

- 公开 GitHub + Zenodo DOI  
- 主张边界、负结果、撤回诚信叙事适合本刊  
- 作者信息、伦理、基金、利益冲突段落已有  
- Fig.1 / E1 / E2 矢量源已具备  

### 投前未就绪（阻塞项）

1. **LoRA 权重公开**（Digital Discovery LLM fine-tuning 条款）  
2. **文稿转入 RSC 模板 + TIFF 图**  
3. **DAS 章节顺序与措辞**按指南改写（禁止 upon request 作为主声明）  
4. **参考文献 RSC 格式**  
5. **APC/OA 协议确认**  
6. Cover letter + preferred reviewers  

---

## 8. 推荐下一步（立刻做）

1. 上传至少 **seed43 LoRA**（最好 N=5）到 Hugging Face 或新建 Zenodo 文件集，拿到可点击 URL/DOI。  
2. 改写 Data availability，去掉 “adapters upon request” 作为唯一途径。  
3. 下载 Article template，开始从 v7 灌稿。  

完成 1–2 后，本稿对 Digital Discovery 的**政策合规**基本闭环；再完成模板与 ScholarOne 即可正式提交。
