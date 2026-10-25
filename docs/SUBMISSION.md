# ORCATHON 2026 — 提交表单填写清单

> 逐项照抄即可。链接处用 `<>` 标记的，替换成你自己的实际地址。

---

## 1. 基本信息

| 字段 | 填写内容 |
| --- | --- |
| **Team Name** | `TerraWatch` （单人参赛，队长即本人） |
| **Project Name** | `TerraWatch — Satellite Surface Change Detection` |
| **Team Size** | `1` |

---

## 2. 链接（三个都是必填，缺一个会被判定为提交不完整）

| 字段 | 填写内容 |
| --- | --- |
| **Live Project Link** | `https://terrawatch-change-detection.app.workbuddy.host/?sharecode=WSQ674JJXR` |
| **GitHub Repository Link** | `https://github.com/ASylviaofmine/terrawatch` |
| **Demo Video Link** | `https://youtu.be/gpFW010gmpE` |

> ✅ **三个链接全部就绪并已验证**：
> - 线上地址：健康检查返回 `{"status":"ok","classes":7}`，首页 200，样本图 200，
>   七个样本的 `/api/detect` 全部返回正确分类（deforestation / urban expansion /
>   excavation / flooding / wildfire / new road / clean 无变化），首页内嵌视频播放器。
> - GitHub：17 次提交，工作树干净，从零 `git clone` 后可直接运行（`CLONE RUNS OK`）。
> - YouTube：已上传为「不公开」，链接可直接播放；App 首页也内嵌了同一个视频。
>
> 📌 **视频的原始文件**仍在仓库里：
> <https://github.com/ASylviaofmine/terrawatch/blob/master/docs/demo.mp4>
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

> TerraWatch was written from scratch for this hackathon. The repository
> contains one new, self-contained Python project; no code from any earlier
> project was copied into it.
>
> The training data is **synthetic** — generated procedurally by
> `src/synth.py`, which is part of this submission. No external dataset is
> used or required.
>
> Method carried over from earlier personal work (disclosed for
> transparency, no source code reused): the discipline of quoting only
> measured numbers in the README, shipping a self-test that proves the
> deployed code path matches the trained one, and treating the demo video as
> a deliverable in its own right.
>
> AI tools were used for drafting parts of the training/evaluation script
> skeletons, the matplotlib figure styling, and the wording of the README.
> The project design, the synthetic data generator, the model, the
> numpy inference implementation, the evaluation harness and the web app
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

## 6. Team Members

| 字段 | 填写内容 |
| --- | --- |
| Name | `<你的姓名>` |
| Email | `<你的邮箱>` |
| Role | `Solo — team leader and sole contributor` |

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

> **Problem statement:** Develop an AI-powered satellite anomaly detection
> system that identifies unusual changes on Earth's surface and distinguishes
> natural anomalies from potential human-made damage.
>
> **Description:** Analyze sample satellite images to detect changes such as
> forest destruction, coastal changes, unusual land excavation, or
> infrastructure damage. Highlight affected areas, compare images captured
> at different times, and generate an automated report.
>
> **Social deliverable:** Image upload, before-and-after comparison,
> highlighted change regions, and a simple report. No physical robots or
> specialised hardware required.

| 赛题要求 | TerraWatch 对应 |
| --- | --- |
| Image upload | 网页端上传 before / after 两张图 |
| before-and-after comparison | 三栏并排：before / after / detected |
| highlighted change regions | 逐像素 7 类着色叠加 |
| distinguish natural vs human-made | 4 类 human-made + 2 类 natural，报告里给出判定 |
| generate an automated report | 自动生成 HTML 报告（面积、置信度、分类表、结论） |
| AI-powered | 5,923 参数的 CNN，训练 3,000 步，评测表在 README |
| No specialised hardware | 纯 CPU + numpy 推理，无 GPU 依赖 |
