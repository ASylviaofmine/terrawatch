# Demo video — 分镜脚本与录制说明

> 目标：**90 秒左右**的成片，1920×1080（当前脚本输出 1280×720，放大即可）。
> 用途：README 嵌入 + 提交表单的 "Video Explanation" 一栏。
> 原则：**每个数字都必须来自模型自己的输出**，不手写、不美化。

---

## 一、最省事的做法：直接用自动生成的视频

```bash
python tools/make_demo.py
```

产出 `docs/demo.mp4`。这段视频里：

- 每一帧的高亮区域都是模型跑出来的（`src/infer.py` 的输出），不是画上去的；
- 每一行的像素数、变化百分比都是模型算出来的；
- 字幕已经写好了英文，不需要配音也能看懂。

**这就是可以直接提交的视频。** 下面是可选的增强版。

---

## 二、想加分：加英文旁白

### 方案 A（推荐）——用仓库自带的录屏脚本

你的 `defect-detection` 仓库里已经有一个可用的 `tools/record_demo.py`（ffmpeg 同录屏幕 +
麦克风，录完自动报告音量 dB）。它不在本仓库里，因为它是通用工具而非本项目代码。
把它复制过来即可：

```bash
cp ../defect-detection/tools/record_demo.py tools/record_demo.py
pip install imageio-ffmpeg
python tools/record_demo.py --list        # 确认能看到麦克风
```

### 方案 B——录网页演示

1. 终端里跑 `python server.py`（或 `python app.py`），确认 <http://127.0.0.1:7860> 能打开。
2. **先热身**：点一次 "Detect changes"，等结果出来，再清空（刷新页面）。
3. 浏览器最大化，`Ctrl` + `=` 放大 2～3 档，让三栏占满画面。
4. 录：
   ```bash
   python tools/record_demo.py demo --seconds 45 --keep-console
   ```
5. 按下面的节奏操作：

| 时间 | 画面 | 旁白 |
| --- | --- | --- |
| 0–6 s | 点 "deforestation" 示例按钮，模型跑完 | "Give it two acquisitions of the same ground…" |
| 6–14 s | 停在结果页，指变化区域 | "…and it marks what changed, and says what kind of change it is." |
| 14–24 s | 点 "flooding" 示例按钮 | "Here the change is natural rather than human-made." |
| 24–34 s | 停在下方的 Changed area / confidence 数字 | "The report gives the changed area and a confidence — both computed, not narrated." |
| 34–45 s | 打开 GitHub 仓库页面 | "The model, the training data generator and the evaluation are all in the repository." |

> ⚠️ **两个最容易翻车的点**：
> 1. 录之前把通知全关（微信 / QQ / 邮件 / 专注助手）。
> 2. 服务**先起好再录**——`numpy` 首次加载 + 权重读取约 1～2 秒，中途启动会出现一段黑屏。

### 字幕

剪映专业版（免费）→ 新建项目 → 导入视频 → 「文本 → 智能字幕 → 识别字幕」→ 改错字 →
导出时勾选「烧录到视频」。旁白音量拉到 -3 dB 左右。

---

## 三、想讲得更深：补充镜头

如果时间允许，加两段会让评委觉得"这个项目是做完的"：

| 镜头 | 画面 | 说什么 |
| --- | --- | --- |
| 混淆矩阵 | `runs/eval/confusion.png` 拖进浏览器 + F11 | "Per-class IoU on 200 held-out scenes. Every class is above X." |
| 自检 | 终端跑 `python tools/selftest.py`（**先跑完再录**，不要现场等） | "The deployed app runs numpy, not PyTorch — and the self-test proves they're the same model to one ten-thousandth." |

> 第二镜的价值在于它回答了评委必然会问的问题："网页上跑的是不是真的你训练的那个模型？"

---

## 四、提交表单要填什么

| 字段 | 内容 |
| --- | --- |
| Project Name | **TerraWatch** |
| Live Project Link | 部署后的线上地址（见 §5） |
| GitHub Repository Link | `https://github.com/ASylviaofmine/terrawatch` |
| Video Explanation | `docs/demo.mp4` 的上传链接（B ��� YouTube Unlisted） |
| Tools Used | Python 3.13 · PyTorch 2.9.1 (CUDA 12.6, RTX 4060) · NumPy · Pillow · Gradio · Matplotlib · Git |
| Team Members | 你的名字 + 邮箱（若单人参赛，填自己） |

---

## 五、部署

```bash
python server.py          # 监听 $PORT，默认 8000
```

`server.py` 只依赖 **numpy + Pillow**，服务的是一个自带的单页 UI（`web/index.html`），
不依赖 Gradio 的版本行为。部署到线上后把地址填进提交表单的 "Live Project Link"。

---

## 六、如果只有 30 分钟

优先级排序（从高到低）：

1. **让 `docs/demo.mp4` 存在** —— `python tools/make_demo.py`，一条命令。
2. **确认线上链接能打开** —— 部署完手动点一次示例按钮。
3. **README 顶部有图、有表、有数字** —— 评委第一眼看的就是这个。
4. 录屏加旁白 —— 有时间再做。

前三条加起来不到 15 分钟，而它们覆盖了"Functionality / Presentation"的大部分分数。
