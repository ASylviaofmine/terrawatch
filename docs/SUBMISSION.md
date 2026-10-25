# ORCATHON 2026 — 提交表单填写清单

> 逐项照抄即可。链接处用 `<>` 标记的，替换成你自己的实际地址。

---

## 1. 基本信息

| 字段               | 填写内容                                              |
| ---------------- | ------------------------------------------------- |
| **Team Name**    | `TerraWatch` （单人参赛，队长即本人）                         |
| **Project Name** | `TerraWatch — Satellite Surface Change Detection` |
| **Team Size**    | `1`                                               |

---

## 2. 链接（三个都是必填，缺一个会被判定为提交不完整）

| 字段                         | 填写内容                                                                           |
| -------------------------- | ------------------------------------------------------------------------------ |
| **Live Project Link**      | `https://terrawatch-change-detection.app.workbuddy.host/?sharecode=WSQ674JJXR` |
| **GitHub Repository Link** | `https://github.com/ASylviaofmine/terrawatch`                                  |
| **Demo Video Link**        | `https://youtu.be/gpFW010gmpE`                                                 |

> ✅ **三个链接全部就绪并已验证**：
>
> - 线上地址：健康检查返回 `{"status":"ok","classes":7}`，首页 200，样本图 200，>     
>   七个样本的 `/api/detect` 全部返回正确分类（deforestation / urban expansion />     
>   excavation / flooding / wildfire / new road / clean 无变化），首页内嵌视频播放器。
> - GitHub：17 次提交，工作树干净，从零 `git clone` 后可直接运行（`CLONE RUNS OK`）。
> - YouTube：已上传为「不公开」，链接可直接播放；App 首页也内嵌了同一个视频。
>
> 📌 **视频的原始文件**仍在仓库里：>   
> <https://github.com/ASylviaofmine/terrawatch/blob/master/docs/demo.mp4>>   
> —— 万一 YouTube 需要重新上传，源文件在，不会丢。

---

## 3. Tools Used（评审会看这一栏有没有如实填写）

```
Python 3.13.15
PyTorch 2.9.1+cu126  (CUDA 12.7, NVIDIA RTX 4060 Laptop 8 GB)
NumPy 2.5.3
Pillow 12.3.0
Gradio 6.28.0
Matplotlib 3.11.2
OpenCV 5.0.0
Git
```

**部署环境（与训练环境不同，如实分开写）：**

```
Python 3.13 · NumPy · Pillow · Python stdlib http.server
（线上服务不依赖 PyTorch / Gradio —— 前向传播用 numpy 重实现，
  由 tools/selftest.py 断言与 PyTorch 模型误差 < 1e-4）
```

---

## 4. Pre-Built Project Disclosure（**必填，且要如实**）

官方规则原文：*Pre-built projects are welcome, provided existing work and  
contributions are clearly disclosed. Misrepresenting existing work as  
entirely new is prohibited.*

**本项目实际状态，照抄这段：**

> TerraWatch was written from scratch for this hackathon. The repository>   
> contains one new, self-contained Python project; no code from any earlier>   
> project was copied into it.
>
> The training data is **synthetic** — generated procedurally by>   
> `src/synth.py`, which is part of this submission. No external dataset is>   
> used or required.
>
> Method carried over from earlier personal work (disclosed for>   
> transparency, no source code reused): the discipline of quoting only>   
> measured numbers in the README, shipping a self-test that proves the>   
> deployed code path matches the trained one, and treating the demo video as>   
> a deliverable in its own right.
>
> AI tools were used for drafting parts of the training/evaluation script>   
> skeletons, the matplotlib figure styling, and the wording of the README.>   
> The project design, the synthetic data generator, the model, the>   
> numpy inference implementation, the evaluation harness and the web app>   
> are my own work.

---

## 5. Project Description（提交表单的描述框）

```
TerraWatch — surface change detection between two satellite acquisitions.

The problem: given two images of the same ground, months apart, a human can
say "the forest is gone" — but not at scale, not with a measurement, and not
with any distinction between human-driven and natural change.

The solution: a small fully-convolutional network (5,639 parameters) that
takes BOTH frames stacked channel-wise and emits 7 per-pixel logits in one
forward pass: no-change, deforestation, urban expansion, excavation, new
road, flooding, wildfire scar. The output is a highlighted change map plus
an automated incident report stating the changed area, the confidence, and
whether the change is human-made or natural.

Measured on 200 held-out synthetic scenes — change IoU, pixel accuracy and
the human-made-vs-natural accuracy are in the README §5, and the README's
results section is generated from runs/eval/metrics.json rather than typed
in, so every number there is the number the evaluation produced.

What makes it credible:
- The training data is procedurally generated with EXACT pixel ground truth,
  so precision/recall/IoU are real measurements rather than impressions.
- The deployment runs numpy only — no PyTorch in the serving path — and
  tools/selftest.py asserts the numpy forward pass matches the PyTorch model
  to < 1e-4, so the hosted demo runs the same model, not an approximation.
- The hosted app is a stdlib HTTP server with a self-contained page; the
  only third-party packages in the serving path are numpy and Pillow.
- The README documents the data bugs the evaluation exposed (excavation and
  flooding first scored IoU 0.03 and 0.00 because their textures were
  indistinguishable from the scene's existing bare soil and water; urban
  expansion later collapsed to 0.07 for the same reason), the fixes, and the
  limits of the synthetic training source.
- The demo shows the negative case: unchanged ground produces no false alarm.
- tools/reproduce.py regenerates every number in the README in one command.

Stack: Python, PyTorch (training), NumPy (inference), Pillow, Matplotlib,
OpenCV. Training: 2,000 steps on an RTX 4060 Laptop GPU.
```

---

## 6. Team Members & Contributions

### 6.1 成员信息（逐格填）

| 字段             | 填写内容                               |
| -------------- | ---------------------------------- |
| Full Name      | `<你的姓名>`                           |
| Email          | `<你的邮箱>`                           |
| Team Role      | `Team Leader — Solo (1 member)`    |
| GitHub         | `https://github.com/ASylviaofmine` |
| Country / City | `<你的国家 / 城市>`                      |

> 📌 赛题允许 **1–3 人**，单人参赛是合规的。**不要为了"看起来像团队"而虚构成员**>   
> —— 成员信息是要核验的，写假名字是硬伤。就写一个人，把贡献写扎实。
>
> 如果表单有 "Team Members" 列表只有一行，就填你自己。

### 6.2 Contributions（贡献说明，直接粘 —— 495/500 字符）

```
I designed, built, trained, evaluated and deployed TerraWatch end to end: the synthetic data generator, the 5,639-parameter change model, numpy-only inference, the evaluation harness and web app. The measurements shaped it: the first evaluation gave excavation IoU 0.03 because its texture matched the bare soil in the scene, so the pairs were ambiguous and I fixed the labels, not the model. AI tools drafted some scripts and docs; the design, model and app are my own work.

```

### 6.3 完整版贡献说明（字数不限时用，或放进 README / 演讲稿）

```
Solo project — I designed, built, trained, evaluated and deployed
TerraWatch end to end.

What I wrote:
- src/synth.py: the procedural satellite-scene generator. Five land-cover
  texture families, six change-event renderers, a per-scene illumination
  and sensor-noise model, and the ground-truth bookkeeping. This is the
  piece that makes every number in the project a measurement rather than
  an impression.
- src/model.py: the 5,639-parameter network, and the dilated third layer
  that widened the receptive field to 15x15 after the evaluation showed
  the model could not see a 40-pixel change region.
- src/infer.py: the numpy re-implementation of the forward pass, so the
  deployed app ships no PyTorch.
- server.py + web/index.html: the deployed app, a stdlib HTTP server
  serving a self-contained page.
- The training loop, the evaluation harness, the confusion matrix, the
  README figures, the demo video and the submission documents.

What the measurements changed:
- The first evaluation returned IoU 0.03 for excavation and 0.00 for
  flooding. The cause was in my data, not the model: those textures were
  identical to the bare soil and the standing water already in the scene,
  so the image pairs were ambiguous by construction. I fixed the labels.
- A later rewrite of the land cover fixed the blocky look and quietly
  broke urban expansion to 0.07 for the same reason. Fixed the labels
  again.
- The numpy port had two shape bugs that made every deployed prediction
  wrong while looking plausible. The self-test now asserts shapes, not
  just values.

AI tools were used for drafting parts of the training/evaluation scripts,
figure styling and README wording. The project design, data generator,
model, inference implementation, evaluation harness and web app are my own
work. This is disclosed in detail in the repository README.
```

### 6.4 为什么这么写

评审看这一栏想知道三件事，这三段都答了：

| 评审的疑问           | 你的回答落在哪                     |
| --------------- | --------------------------- |
| 这活儿到底谁干的？       | 第一句"端到端由我完成"，且列出了具体模块       |
| 工作量有多少？         | 逐个模块列出，并给出"改变了什么"的三个真实案例    |
| 是不是套壳 / AI 生成的？ | 主动披露 AI 辅助范围 + 三个可验证的数据缺陷案例 |

第三点尤其重要：**主动说"AI 帮我起草了脚本和文档，但设计和实现是我的"，  
比被评审问出来要好得多。** 规则要求披露预构建工作，隐瞒的风险远大于坦白的风险。

---

## 7. 提交前的最后检查清单

- [x] 线上地址已部署并验证（健康检查 / 首页 / 样本图 / 七个样本的检测全部通过）
- [x] GitHub 仓库是 Public，评委不需要登录
- [x] `runs/weights.npz` 已提交（否则别人克隆下来跑不起来）
- [x] README 首屏有：图、表、结论、一句话数据
- [x] README 里的每个数字都能在仓库里找到出处（`runs/metrics.json`、`runs/eval/metrics.json`）
- [x] Pre-Built Disclosure 已如实填写
- [x] **视频已上传 YouTube（不公开），链接 `https://youtu.be/gpFW010gmpE`**
- [ ] **提交前用无痕窗口点一遍三个链接**（这一步 30 秒，能挡掉 90% 的意外）
- [ ] 提交页所有字段都填了 —— **空字段 = 提交被判不完整 = 不进评审**
- [ ] 赛题 track 选对了（SpaceTech & Satellite Intelligence）

### 提交时逐格粘什么

```
Project Name                     → tools/form_text.py 的第 1 段
Problem Statement                → 第 2 段
Live Project Link                → 第 3 段
GitHub Repository Link           → 第 4 段
Demo Video Link                  → 第 5 段
Technologies and AI Tools Used   → 第 6 段
Pre-Built Disclosure             → 第 7 段
Project Description              → 第 8 段
Team Name / Members / Email      → 你的信息
```

`python tools/form_text.py` 会一次打印全部八段，并标出每段的字符数  
（都 ≤ 500，页面右下角也显示 "Remaining characters"）。

### 如果只剩 15 分钟

按这个顺序做，每一步都只要一两分钟：

1. 打开 Unstop 提交表单，逐格粘贴。
2. 复制 `docs/SUBMISSION.md` §5 的描述文本粘贴进描述框。
3. Tools Used 复制 §3 的两段。
4. Pre-Built Disclosure 复制 §4。
5. **逐格确认没有空格**，点提交。

**不要在这个环节改代码。** 代码已经能跑、线上已经能开、视频已经传好。  
最后这段时间改任何一行代码都是给自己制造风险。

---

## 8. 赛题原文对照（确认没跑偏）

> **Problem statement:** Develop an AI-powered satellite anomaly detection>   
> system that identifies unusual changes on Earth's surface and distinguishes>   
> natural anomalies from potential human-made damage.
>
> **Description:** Analyze sample satellite images to detect changes such as>   
> forest destruction, coastal changes, unusual land excavation, or>   
> infrastructure damage. Highlight affected areas, compare images captured>   
> at different times, and generate an automated report.
>
> **Social deliverable:** Image upload, before-and-after comparison,>   
> highlighted change regions, and a simple report. No physical robots or>   
> specialised hardware required.


| 赛题要求                              | TerraWatch 对应                        |
| --------------------------------- | ------------------------------------ |
| Image upload                      | 网页端上传 before / after 两张图             |
| before-and-after comparison       | 三栏并排：before / after / detected       |
| highlighted change regions        | 逐像素 7 类着色叠加                          |
| distinguish natural vs human-made | 4 类 human-made + 2 类 natural，报告里给出判定 |
| generate an automated report      | 自动生成 HTML 报告（面积、置信度、分类表、结论）          |
| AI-powered                        | 5,923 参数的 CNN，训练 3,000 步，评测表在 README |
| No specialised hardware           | 纯 CPU + numpy 推理，无 GPU 依赖            |
