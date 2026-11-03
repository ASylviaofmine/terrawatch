# Erasmus Mundus CV — 英文学术版内容

> 目标：**2 页**（IFRoS 明确要求 CV ≤ 2 页）。这是内容稿，排版时用 consortium
> 提供的模板（多数 EMJMD 有自己的 .docx 模板，**必须用他们的**，否则格式会被扣分）。
>
> ⚠️ **你现在这份 `李海钰 个人简历(1).docx` 不能用。** 它是中文的、面向
> 网站/SEO/内容运营实习的（意向岗位写的是运营），Erasmus 要的是英文**学术 CV**。
> 但它的内容可以复用：公众号运营、跑步社社长、Excel 数据复盘，这些在 Erasmus
> CV 里对应 "Communication / Organisation / Data analysis"，不是没用。

---

## 排版提示

- **一页 = 一栏**，不要双栏（欧洲学术 CV 惯例）
- 页边距 2cm，字号 10.5–11pt，行距 1.15
- 姓名用 **Haiyu Li**（不是 "Haiyu Li Li"，见 `docs/CERTIFICATE.md`）
- 所有日期用 `Oct 2026` / `Sep 2022 – Jun 2027` 这种英文月份缩写
- **不要放照片**（EMJMD 明确不要，会被扣分）

---

# 页面 1

## HAIYU LI

Jinling Institute of Technology (金陵科技学院) · Nanjing, China
Student ID 2217011031 · Bachelor of Engineering, Automation (2022.09 – 2027.06)
`asylviaofmine@outlook.com` · `github.com/ASylviaofmine` · LinkedIn: `linkedin.com/in/asylviaofmine`

---

### EDUCATION

**Jinling Institute of Technology** — B.Eng. Automation Engineering, Nanjing, China
Sep 2022 – Jun 2027 *(expected)*

- Core coursework: Automatic Control Principles, Digital & Analog Electronics,
  Microcomputer Principles and Applications, Mathematics Modelling
- **GPA 2.91 / 4.0** (mean 82.09/100), 145 ECTS-equivalent credits completed
- Academic ranking: **96/100 in Mathematical Modelling**, National Mathematical
  Contest in Modelling (Honourable Mention, team 2516158)

> 📌 **如果你的排名更好看，换成排名。** GPA 2.91 不算亮眼，但「数学建模 96 分」
> 和数学建模竞赛 H 奖是硬的。如果大三下有 GPA 峰值（3.3+），写 "GPA 3.x/4.0
> (best semester)" 比写平均分强。

---

### SELECTED PROJECTS

**TerraWatch — Satellite Surface Change Detection** `github.com/ASylviaofmine/terrawatch`
*Individual research project, Oct 2026*

Deep-learning system that takes two satellite acquisitions of the same ground
months apart and produces a per-pixel 7-class change map, distinguishes
human-made change (deforestation, urban expansion, excavation, new road) from
natural change (flooding, wildfire scar), and writes an automated incident
report.

- Designed and implemented the **procedural remote-sensing data generator**
  with exact pixel-level ground truth, making every reported metric a
  measurement rather than an estimate
- Trained a 5,639-parameter fully-convolutional network (dilated third
  convolution, 15×15 receptive field); **change IoU 0.905, pixel accuracy
  99.3%, 88.1% accuracy at separating human-made from natural change**, measured
  on 200 scenes generated from a seed range disjoint from training
- **Diagnosed three label-level data defects by measurement** (excavation,
  flooding and urban expansion recovered from IoU 0.03 / 0.00 / 0.07 to
  0.57 / 0.64 / 0.61), showing the failures stemmed from class separability
  in the data rather than model capacity
- Re-implemented the forward pass in **NumPy only** for deployment and shipped a
  self-test asserting agreement with the PyTorch model to < 1e-4
- Deployed a live web application (Python standard-library HTTP server, no
  framework in the serving path) and a 22-second captioned demo video

**Nemotron-Supervised A\* + PID Navigation Agent** `github.com/ASylviaofmine/nemotron-nav-agent`
*Individual project*

Differential-drive robot that plans with A\* on an occupancy grid and tracks
the path with a proportional controller, under the supervision of an NVIDIA
Nemotron model served from Nebius Token Factory — i.e. a VLM driving a
planning-and-control loop rather than a scripted demo.

**Steel Surface Defect Detection with YOLOv8** `github.com/ASylviaofmine/defect-detection`
*Individual project*

YOLOv8 detector for steel-surface defects, benchmarked against a
classical-CV baseline that was deliberately implemented first, so the
learning-curve comparison is defensible rather than assumed.

**2D Mobile-Robot Navigation Simulator** `github.com/ASylviaofmine/robot-nav-sim`
*Individual project*

Grid-based path planning with feedback control, written from scratch with no
robotics framework, used as the substrate for the navigation work above.

> 📌 **每个项目给 2–4 条 bullet，最后一条必须带数字或硬事实。** "从零实现"本身
> 不算证据，"无框架从零实现 + 与 YOLOv8 做了同数据基线对比"才算。

---

## 页面 2

### COMPETITIONS & AWARDS

| Competition | Track / Category | Date | Result |
| --- | --- | --- | --- |
| **ORCA CODERS: ORCATHON 2026** (Dev Harbour, online, 654 participants) | 05 SpaceTech & Satellite Intelligence | Oct 2026 | **Certificate of Participation** (valid submission) |
| Chinese Mathematics Competitions in Higher Education (华数杯) | National final | 2025 | **First Prize** |
| Mathematical Contest in Modelling (MCM/ICM, USA) | H award | 2025 | **Honourable Mention** (team 2516158) |
| International English Vocabulary Competition | — | 2024 | First Prize |
| Debate Competition, School of Intelligent Control | — | 2024 | 4th place |

- ORCATHON 2026 project **TerraWatch** — source code, live deployment and demo
  video publicly available (`github.com/ASylviaofmine/terrawatch`)

---

### TECHNICAL SKILLS

**Programming** Python (NumPy, PyTorch, OpenCV, Matplotlib, scikit-learn),
C/C++ (basics), SQL, Bash
**Frameworks & tools** PyTorch, Gradio, Git/GitHub, Linux, ROS (fundamentals),
Docker (basics)
**Domains** Computer vision, remote-sensing image analysis, deep learning for
per-pixel semantic segmentation, robot path planning (A\*, PID), multi-agent
simulation, natural-language processing with LLM APIs
**Data** Synthetic data generation with pixel-exact ground truth, evaluation
protocol design (held-out seed ranges, IoU / precision / recall / F1), confusion
matrix analysis, Excel for statistical analysis and reporting

---

### LANGUAGES

| Language | Level |
| --- | --- |
| Chinese (Mandarin) | Native |
| English | Professional working proficiency — **CET-4**; academic reading and writing (technical papers, documentation) |

> ⚠️ **这一栏现在是你最大的短板，必须改。**
> IFRoS 要求 B2 英语，E-PiCo+ 明确只认 **IELTS 6.5 / TOEFL iBT 80 /
> Cambridge B2 173 / TOEIC 800**，且"母语学位免交成绩"的豁免对你不适用
> （你的学位是中文授课）。
>
> **在拿到成绩前，这一栏不能这么写。** 考出成绩后改成：
> `English — IELTS 6.5 (band scores: L 6.5 / R 7.0 / W 6.0 / S 6.0)`，
> 分数高就写具体分数。**这比任何竞赛奖项都重要**，因为没语言成绩是直接被刷，
> 奖项只是加分。
>
> 在此之前，如果必须现在投，先考一个 **Cambridge B2 First 173** 或 **TOEIC 800**——
> 备考周期比雅思短，认可度对 EMJMD 是一样的。

---

### LEADERSHIP & ACTIVITIES

- **President, University Running Club** — Sep 2023 / Jun 2025 · organised 4
  campus events; managed 80+ members
- **Media Account Lead** — Sep 2023 / present · owned content strategy, SEO
  topic research and audience analysis; grew to 700+ followers with a peak
  article at 55,000 views and a 14% follow-conversion rate; tracked
  performance in Excel to iterate on each cycle
- **Volunteer, World Youth Forum (Outstanding Volunteer)** · VR Industry
  Expo 2026 volunteer · half-month volunteer service at the Leiyin Temple,
  Dunhuang

---

### 加分项：单独的一栏（可选）

很多 EMJMD 申请表有一栏 **"Additional Information"** 或允许附项目链接。
在这里放：

```
GitHub:  github.com/ASylviaofmine
         Four self-contained projects with full documentation, evaluation
         harnesses and reproducible results:
         terrawatch (satellite change detection, IoU 0.905),
         nemotron-nav-agent (A* + PID with LLM supervision),
         defect-detection (YOLOv8 steel defect detection),
         robot-nav-sim (from-scratch A* navigation simulator)
         Each repository includes a self-test that verifies the deployed
         code path matches the trained model.
```

---

## 常见问题速查

| 问题 | 答案 |
|---|---|
| 参赛证书要不要附 PDF？ | **一般不附**。CV 里写一行 + 证书在 "Additional" 栏注明。除非表单明确要 "proof of awards"。 |
| ORCATHON 只有参赛奖，写出来会不会显得弱？ | **不会**，因为你写的是项目不是证书。CV 里 TerraWatch 的权重远大于那一行参赛记录。 |
| 四个项目会不会太多？ | EMJMD 审 CV 主要看"技术能力+独立完成"的证据。四个都留，但 **TerraWatch 详细写、其余三个各 2 条**。 |
| 公众号要不要写？ | 写，但只留 1 条并**改成能力语言**（内容策略、SEO、Excel 数据复盘）。目标是"沟通与组织能力"，不是"我做过自媒体"。 |
| GPA 2.91 会不会被扣分？ | 会，所以要用数学建模 96 分和竞赛成绩做补充。但**不要改数字或隐瞒**——EMJMD 背景调查会核对。 |
| 要不要写 "AI-assisted"？ | **不要写在 CV 上**。CV 是成果陈述。AI 披露放在项目仓库 README 里（已经做了）。若被问到再如实说明。 |
