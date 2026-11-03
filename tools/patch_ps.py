"""Patch docs/PS_ERASMUS.md with the constraints read off the IFRoS site.

The site states the motivation letter "cannot be longer than a single
page" and explicitly invites a 30-60s video link. The draft written
earlier assumed 1-2 pages, so the header, the IFRoS draft, the checklist
and the timeline all need correcting.

Run once; it is idempotent.
"""

from pathlib import Path

P = Path(__file__).resolve().parent.parent / "docs" / "PS_ERASMUS.md"
t = P.read_text(encoding="utf-8")

# ---- 1. header: the real constraints -------------------------------------
old_head = """> 长度：**1–2 页**（IFRoS 明确写 "Motivation letter (1–2 pages) – tell your robot story"）。
> 每个 consortium 有自己的模板，**先下载模板再往里填**。"""

new_head = """> ⚠️ **IFRoS 官网原文对 PS 的硬性要求是「不能超过一页」**
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
>   ② **可点开的链接/出处**；③ **这些成果如何塑造了你对这个项目的偏好**。"""

assert old_head in t, "header not found"
t = t.replace(old_head, new_head)

# ---- 2. the IFRoS draft: single page, four achievements ------------------
start = t.index("## 三、IFRoS 版本")
end = t.index("## 四、通用版")
new_ifros = """## 三、IFRoS 版本（单页，约 480 词）— 可直接用

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

"""
t = t[:start] + new_ifros + t[end:]

# ---- 3. checklist ------------------------------------------------------
t = t.replace(
    "- [ ] 长度符合要求（IFRoS：1–2 页；E-PiCo+：mandatory template）",
    "- [ ] 长度符合要求（**IFRoS：单页**；E-PiCo+：mandatory template）\n"
    "- [ ] **至少 3 个**与该方向相关的成果，且**每个都带可点开的链接**\n"
    "- [ ] 写了\"这些成果如何塑造了你的选择\"（官网明文要求）\n"
    "- [ ] 附了 **30–60 秒视频链接**（IFRoS 明确欢迎）\n"
    "- [ ] 确认 PS 里的链接**点开能跑**（评审会点）",
)
t = t.replace(
    "- [ ] 附了 GitHub 链接，并确认**点开能跑**（IFRoS 明确欢迎 \"GitHub link\"）\n", ""
)
t = t.replace(
    "- [ ] 附了推荐信（IFRoS 要两封，**提前 3 周找老师要**， professors 放假就晚了）",
    "- [ ] 推荐人在系统里填好了**机构邮箱**（他们会自己收到邮件并提交）",
)

# ---- 4. timeline --------------------------------------------------------
t = t.replace(
    "| **本周** | 报名英语考试并**约考位** | IFRoS / E-PiCo+ 都硬性要求 IELTS 6.5 / TOEFL 80 / Cambridge B2 173 / TOEIC 800。四选一，但**考位紧张，越早约越好** |",
    "| **本周** | **约雅思机考考位** + 定下两位推荐人并发信 | 语言是唯一硬门槛；老师要赶在放寒假前 |",
)
t = t.replace(
    "| **11 月内** | 找两位推荐人（专业课老师 + 竞赛指导老师）并说明项目 | 老师放寒假就来不及了 |",
    "| **11 月中** | 教务处开**在读证明 + 居住地证明 + 成绩单英文翻译** | 一趟办完三件事 |\n"
    "| **11 月底** | 查**护照到期日**（须晚于毕业年 12 月） | 不够就马上换 |",
)
t = t.replace(
    "| **12 月** | 写 CV + PS + 申请各项目 | 每份 PS 都要改课程名，**不要想着一稿通投** |",
    "| **12 月上旬** | **考雅思机考**（3–5 天出分） | 出分后还剩 6 周处理文书 |\n"
    "| **12 月** | 写 CV + PS，每个项目单独改课程名 | **不要一稿通投** |",
)
t = t.replace(
    "| **2027-01-27** | DENSYS2.0 奖学金截止 | |",
    "| **2027-01-20** | **所有奖学金项目统一投完** | 集中 1 月 31 日截止，别卡最后一天 |\n"
    "| **2027-01-27** | DENSYS2.0 奖学金截止 | |",
)
t += """
> 📌 **IFRoS 的 2027–2029 申请还没开放**（官网原文："Applications for the
> 2027–2029 intake are not open yet. The application calendar, deadlines and
> information on available funding opportunities will be published on this page
> as soon as they are confirmed."）→ **定期回看官网页面 / 订阅邮件**，一开放
> 就动手。**别等它开放了才开始写文书。**
"""

P.write_text(t, encoding="utf-8")
print("patched docs/PS_ERASMUS.md")
print("  header  :", "single page" in t or "不能超过一页" in t)
print("  ifros   :", "480 词" in t)
print("  video   :", "30s–60s" in t)
print("  length  :", len(t), "chars")
