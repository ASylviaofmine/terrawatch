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
| **Live Project Link** | `<部署后的线上地址>` |
| **GitHub Repository Link** | `https://github.com/ASylviaofmine/terrawatch` |
| **Video Explanation Link** | `<docs/demo.mp4 上传后的链接（B 站 / YouTube Unlisted）>` |

> ⚠️ **提交前逐个点开这三个链接。** 提交表单要填的是能打开的地址，不是"应该能打开"的地址。
> 如果线上部署来不及，先用 `https://github.com/ASylviaofmine/terrawatch` 里的
> `docs/demo.mp4` 顶上去，同时在表单的 Description 里说明线上地址将补上——
> 但**优先把线上部署搞定**，因为 "Live Project Link" 是评审清单上的独立一项。

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
say "the forest is gone" — but not at scale, not with a measurement, and
not with any distinction between human-driven and natural change.

The solution: a small fully-convolutional network (5,923 parameters) that
takes BOTH frames stacked channel-wise and emits 7 per-pixel logits in one
forward pass: no-change, deforestation, urban expansion, excavation, new
road, flooding, wildfire scar. The output is a highlighted change map plus
an automated incident report that states the changed area, the confidence,
and whether the change is human-made or natural.

What makes it credible:
- The model is trained on procedurally generated scenes with EXACT pixel
  ground truth, so every number in the README is measured, not estimated.
- The deployment runs numpy only — no PyTorch in the serving path — and
  tools/selftest.py asserts the numpy forward pass matches the PyTorch model
  to < 1e-4, so the hosted demo runs the same model, not an approximation.
- The demo also shows the negative case (unchanged ground) and the
  limitations are stated explicitly in the README rather than omitted.

Stack: Python, PyTorch (training), NumPy (inference), Pillow, Gradio,
Matplotlib, OpenCV. Training: 3,000 steps on an RTX 4060 Laptop GPU.
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

- [ ] 三个链接逐个点开，都能打开
- [ ] `docs/demo.mp4` 已上传，且**视频里有声音或字幕**（评委不会静音看图）
- [ ] README 首屏有：一张图、一张表、一句结论
- [ ] README 里的每个数字都能在仓库里找到出处（`runs/metrics.json`、`runs/eval/metrics.json`）
- [ ] Pre-Built Disclosure 已如实填写 —— **这一栏被抓到造假会直接取消资格**
- [ ] 仓库是 Public，评委不需要登录
- [ ] `runs/weights.npz` 已经提交（否则别人克隆下来跑不起来）
- [ ] 赛题要求的 track 已选对（SpaceTech & Satellite Intelligence）
- [ ] 在提交页确认所有字段都填了 —— **空字段 = 提交被判不完整 = 不进评审**

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
