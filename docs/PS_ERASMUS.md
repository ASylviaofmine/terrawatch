# Erasmus Mundus Personal Statement / Motivation Letter — 写法与全文

> ⚠️ **IFRoS 官网原文对 PS 的硬性要求是「不能超过一页」**
> （*"The motivation letter cannot be longer than a single page."*）。
> 下面的 IFRoS 版就是按单页设计的（约 480 词）。E-PiCo+ 用 **mandatory template**
> （他们自己的 .docx），不是自由格式。
>
> ⚠️ **官网还有一条对你极其有利**：
> *"You can also include a link to a **30s–60s video** explaining your projects
> related to the master."*
> → 你现在的演示视频 22 秒，**建议另剪一版 45–60 秒**（把结尾的实测数字条形图
> 换成"三个失败-修复案例"的快切）。现有版本可以先用，链接随时可换。
>
> **官网对 PS 内容的原话要求**（照着写）：
> *"The letter should clearly state (**preferably 3 or more**) achievements of
> your past curriculum which are related to the area of **Field Robotics** with
> clear references or links to support papers, courses or projects, and
> **how these shaped your preferences** towards this master program."*
>
> → 三件事缺一不可：① **至少 3 个**与 Field Robotics 相关的成果；
>   ② **可点开的链接/出处**；③ **这些成果如何塑造了你对这个项目的偏好**。

---

## 一、核心原则：证书放 CV，项目放 PS

这是最容易搞错的地方。

| 放哪 | 放什么 | 为什么 |
|---|---|---|
| **CV** | 「ORCATHON 2026 — Certificate of Participation」**一行** + TerraWatch 完整条目 | CV 是**事实清单**，评审快速扫，量化结果比奖项名称有说服力 |
| **PS / Motivation Letter** | **不出现**"I participated in a hackathon"这种句式 | PS 是**动机叙事**。参赛证书在 PS 里是负资产——它暗示"我只是参加了一下" |
| **PS 里真正该写的** | 项目本身 + 从中获得的**方法论**，证书最多作为一句佐证出现 | 评审要的是"你为什么适合做研究"，不是"你拿过什么奖" |

**一句话心法**：**证书是弱证据，项目是强证据。** CV 用项目撑起重量，证书做点缀；
PS 让项目讲故事，证书最多被提到一次。

### 最容易犯的错（改之前先看）

```
❌ I participated in ORCATHON 2026 and won a participation certificate.
   This experience taught me teamwork and perseverance.
```

```
✅ In October 2026 I built a satellite change-detection system in a single
   working day and shipped it: a deployed web application, a captioned demo
   video, and a public repository. What that day actually taught me was how
   to be wrong in a measurable way — and how to find out.
```

第二段的每一个句子都在推进"我具备做研究的能力"，没有一个字在报奖。

---

## 二、选哪个项目当主线

| 申请方向 | 主线项目 | 理由 |
|---|---|---|
| **机器人方向**（IFRoS / MIR / DENSYS / META4.0）| **nemotron-nav-agent** 或 **robot-nav-sim** | IFRoS 官方申请建议就是 "Explain one robot project you did"。用 TerraWatch 讲"感知与 AI"，但主线要留给机器人 |
| **AI / 感知 / 数据方向** | **TerraWatch** | 有最硬的数据、量化的 IoU、可验证的公开仓库 |
| **能源 / 控制 / 电气方向**（E-PiCo+ / EU-CORE / STEPS / DCMAPS / RESCO）| **不要硬塞 TerraWatch** | 匹配度太低，硬写会显得跑题。见下面 §5 |

---

## 三、IFRoS 版本（单页，约 480 词）— 可直接用

> 官网要求 "**preferably 3 or more** achievements related to Field Robotics"，
> 所以这段明确点了**四个**项目，每个都带可点开的链接。

```
Dear Members of the Selection Committee,

I am applying to the Erasmus Mundus Joint Master in Intelligent Field
Robotic Systems because I want to learn how to build systems that fail
safely in the physical world — a problem I have so far only been able to
study in software.

Four projects have shaped that preference, and all four are public.

The one that changed how I work is a wheeled-robot navigation agent
(github.com/ASylviaofmine/nemotron-nav-agent). The robot plans a path with
A* on an occupancy grid and tracks it with a proportional controller, while
an NVIDIA Nemotron model running as a language model supervises the loop
instead of a hand-written script deciding what to do next. Building it
taught me that the hard part is not the planning algorithm but the
interface between what a model proposes and what a controller can safely
execute — which is exactly the seam this programme is built to study.

Underneath it I wrote a 2D mobile-robot navigation simulator from scratch
(github.com/ASylviaofmine/robot-nav-sim): grid path planning and feedback
control, with no robotics framework underneath, because I wanted to
understand the algorithms rather than call them. A third project taught me
the other half of the discipline: a YOLOv8 detector for steel-surface
defects (github.com/ASylviaofmine/defect-detection), benchmarked against
a classical-CV baseline I implemented deliberately first, so the comparison
would be defensible rather than assumed.

The fourth is further from robotics but shaped me most. TerraWatch is a
satellite surface-change detection system I built alone
(github.com/ASylviaofmine/terrawatch, with a live deployment and a demo
video). It ships a procedurally generated dataset with exact pixel-level
ground truth, a 5,639-parameter fully-convolutional network, and an
evaluation harness that measures change IoU (0.905) and accuracy at
separating human-made from natural change (88.1%) on 200 held-out scenes.
Its first evaluation failed: two of six classes came back at IoU 0.03 and
0.00, and the obvious response — a larger network — would have been the
wrong one. Those classes had textures identical to surfaces already in the
scenes, so the image pairs were genuinely ambiguous and no architecture
could separate them. Fixing the data rather than the network took them to
0.57 and 0.64. I now measure first and hypothesise second.

That habit is why I want IFRoS specifically. The consortium's progression —
foundations in manipulation, localisation, perception and machine learning
at Girona, a specialisation year in Zagreb or Budapest, then an
independently supervised thesis — is the shape I have been building for
myself, and it is the shape I want to be pushed through by people who know
more than I do. Every project above was built and published by me alone,
which means I am used to owning a problem end to end and to being honest
about which parts are unfinished. I would welcome the collaboration and
the harder problems a cohort across three countries will bring.

My project work, evaluation harnesses and reproducible results are public
at github.com/ASylviaofmine. A short demonstration video of the
navigation and change-detection projects is available at
youtu.be/gpFW010gmpE. I would be glad to discuss any of it in person.

Yours sincerely,
Haiyu Li
```

**逐条对应官网要求**

| 官网要求 | 这段哪里满足 |
|---|---|
| ≥3 个与 Field Robotics 相关的成果 | 四个项目全列，且开头写 "all four are public" |
| clear references 或 links | 四条 github.com 链接 + 视频链接 |
| 这些如何塑造了对该项目的偏好 | 每段结尾一句 "taught me…" / "shaped me most" / "That habit is why I want IFRoS" |
| 单页 | 约 480 词 |

---

## 四、通用版（AI / 感知 / 数据方向主线 = TerraWatch）

```
Dear Members of the Selection Committee,

I am applying to [PROGRAMME] because I want to work on the problem of
reading change from data — and I want to learn to do it at a level where
the answer has to be defensible.

In October 2026 I participated in a four-hour satellite-imaging hackathon,
and I spent the following weeks rebuilding what I had made. The version I
am submitting measures a change IoU of 0.905 and separates human-made
from natural change with 88.1% accuracy on 200 held-out scenes. The
technique is a 5,639-parameter fully-convolutional network; the part I care
about is the evaluation.

The first evaluation failed. Two of six change classes came back at IoU
0.03 and 0.00, and the natural conclusion was that the network was too
small. It was not. Those two classes had textures that were identical to
surfaces already present in the same scenes, so the image pairs were
ambiguous by construction — a fresh excavation and a patch of dry soil are
the same pixels. No architecture could separate them. Rebuilding the
labelling so each class had a texture that could not be confused with
anything else took them to 0.57 and 0.64, and a later change to the data
broke a third class in exactly the same way, which confirmed the
explanation. Everything I now know about evaluating a model, I learned
from a metric that refused to improve.

What I want from [PROGRAMME] is the environment in which to take that
method seriously. My work so far has been self-directed and its gaps are
obvious: it has never seen a real satellite acquisition, it assumes the two
images are perfectly registered, and it models neither clouds nor shadows.
I have written all of these limitations down rather than around them, and
they are the specific reasons I want to study where the data is real and
the supervision is rigorous. The course structures in [PROGRAMME] —
particularly [SPECIFIC COURSES] — are where I want to close those gaps.

I work alone by default: all four of my projects, including the change
detection system, its procedurally generated dataset, the NumPy-only
deployment path and the evaluation harness, are built and published from
scratch. I am applying as an individual applicant. I would like to be
pushed by a cohort that disagrees with me, and I would rather be the
person in the room who asks whether the measurement actually supports the
claim.

My project work is public at github.com/ASylviaofmine, with the live
system, the evaluation code and the reproducible metrics. I would be glad
to walk through any of it.

Yours sincerely,
Haiyu Li
```

---

## 五、电气/能源方向（E-PiCo+ 等）怎么写

**不要硬塞 TerraWatch。** E-PiCo+ 是电机、功率电子、电池和电驱控制，
TerraWatch 是感知与 AI，评审一眼看出不相关。

**正确做法**：用 `defect-detection`（YOLOv8，机器视觉 + 工业质检，属于
"智能制造"叙事）+ 自动化专业的控制课背景，构造一条"感知—控制—工业落地"
的线。竞赛部分提 ORCATHON 但只作为"参与过国际工程赛事"的背景，不展开项目细节。

```
I am applying to E-PiCo+ because the shift it asks for — from designing
systems to controlling them in motion — is the shift my own projects have
been moving towards for two years.

My most recent work is a YOLOv8 detector for defects on steel surfaces,
built to be judged against a classical computer-vision baseline that I
implemented deliberately first, so the comparison would be honest rather
than assumed. Industrial defect detection is, in the end, a control
problem: you are deciding, for a line moving at a known speed, which
observations can be ignored and which require the process to stop. That is
the same question E-PiCo+ asks in a different vocabulary, and I want to
learn it from people who work on the physics rather than only on the
signal.

My degree is in Automation Engineering, which gave me control theory,
microcomputer interfacing and an electronic machines course; my practical
work has been in machine vision, where I have had to learn both what a
detector can be trusted to see and what a controller should do when it is
not. I am applying as an individual applicant, and I am conscious that
I am arriving from the perception side of the field rather than the
electro-mechanical one — that is a gap, and it is the reason I want the
programme's propulsion and control coursework rather than another
course in detection.

In October 2026 I also took part in an international engineering
hackathon, building a satellite change-detection system end to end within a
single working day and deploying it publicly. The exercise taught me to
scope what can genuinely be finished in a fixed window, and that the
difference between a demonstration and a deliverable is mostly in the
error handling.

My project work is public at github.com/ASylviaofmine, including the
evaluation harness behind the defect-detection results. I would be glad to
discuss the work in detail.

Yours sincerely,
Haiyu Li
```

---

## 六、必须改的占位符

| 占位符 | 怎么填 |
|---|---|
| `[PROGRAMME]` | 项目全名，**照抄官网**（例如 "Erasmus Mundus Joint Master in Intelligent Field Robotic Systems"）|
| `[SPECIFIC COURSES]` | **至少 2 门该项目的真实课程名**。这是最容易露馅的地方——写不出来说明你没研究过 |
| 结尾的邮箱/签名 | `Haiyu Li`（不是 "Haiyu Li Li"）|

---

## 七、提交前自查

- [ ] **用了 consortium 提供的模板**（不是自己排的）
- [ ] 长度符合要求（**IFRoS：单页**；E-PiCo+：mandatory template）
- [ ] **至少 3 个**与该方向相关的成果，且**每个都带可点开的链接**
- [ ] 写了"这些成果如何塑造了你的选择"（官网明文要求）
- [ ] 附了 **30–60 秒视频链接**（IFRoS 明确欢迎）
- [ ] 确认 PS 里的链接**点开能跑**（评审会点）
- [ ] 提到了**具体课程名**（≥2 门）和**具体项目结构/城市**
- [ ] **姓名是 Haiyu Li**，和证书、成绩单、护照一致
- [ ] 至少一个**具体数字**（IoU 0.905 / 200 scenes / 5,639 parameters / 4.5h）
- [ ] 有一个**真实的失败与修复**（这是最能区分"会说"和"真做过"的内容）
- [ ] **没有**"I am a hard worker and a team player"这类空话
- [ ] 语气是"我为什么想做这个"，不是"我做过什么"（后者是 CV 的活）
- [ ] **没有**在 PS 里说 "AI-assisted"（披露放仓库 README，PS 是你的论述）
- [ ] 推荐人在系统里填好了**机构邮箱**（他们会自己收到邮件并提交）

---

## 八、时间线（按今天 2026-11-02 算）

| 什么时候 | 做什么 | 为什么是这个点 |
| --- | --- | --- |
| **本周** | **约雅思机考考位** + 定下两位推荐人并发信 | 语言是唯一硬门槛；老师要赶在放寒假前 |
| **11 月内** | 拿到语言成绩 | 你的 CET-4 不被认可。这是**唯一的硬门槛**，其余都能补救 |
| **11 月中** | 教务处开**在读证明 + 居住地证明 + 成绩单英文翻译** | 一趟办完三件事 |
| **11 月底** | 查**护照到期日**（须晚于毕业年 12 月） | 不够就马上换 |
| **12 月上旬** | **考雅思机考**（3–5 天出分） | 出分后还剩 6 周处理文书 |
| **12 月** | 写 CV + PS，每个项目单独改课程名 | **不要一稿通投** |
| **12 月底** | 把 ORCATHON 证书的姓名问题解决掉 | 姓名不一致会被要求补材料 |
| **2027-01-20** | **所有奖学金项目统一投完** | 集中 1 月 31 日截止，别卡最后一天 |
| **2027-01-27** | DENSYS2.0 奖学金截止 | |
| **2027-01-31** | E-PiCo+ / EU-CORE / IMFSE / MULTIPHASE / SARENA / RESCO / DENSYS 奖学金截止 | 集中在这天，**建议统一 1 月 20 日前全部投完** |
| **2027-04-30** | 自费截止 | 留作补投窗口 |

> ⚠️ **语言成绩这一条比任何文书都优先。** 缺语言成绩 = 直接被刷，
> 写得再好的 PS 也救不了。**先约考位，再写文书。**

> 📌 **IFRoS 的 2027–2029 申请还没开放**（官网原文："Applications for the
> 2027–2029 intake are not open yet. The application calendar, deadlines and
> information on available funding opportunities will be published on this page
> as soon as they are confirmed."）→ **定期回看官网页面 / 订阅邮件**，一开放
> 就动手。**别等它开放了才开始写文书。**
