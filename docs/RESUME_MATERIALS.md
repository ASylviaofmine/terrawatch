# ORCATHON 2026 参赛记录 — 简历与申请材料文案

> 三种格式，按场景挑一个用。所有数字都来自 `runs/eval/metrics.json`，
> 和仓库 README 里那张表是同一份数据。
>
> ⚠️ **官方证书上的姓名是「Haiyu Li Li」（姓重复），需要更正。**
> 详见 `docs/CERTIFICATE.md`。**提交材料前务必先处理这件事** ——
> Erasmus 申请表要求姓名与护照/成绩单一致。
> 下面文案统一用正确的 **Haiyu Li**（名 Haiyu，姓 Li）。

**已核实的事实**

| 项 | 值 |
|---|---|
| 证书 | Certificate of Participation，2026-10-26 签发 |
| 赛事 | Orca Coders: Orcathon（DevHarbour 主办，Unstop 平台，654 人参赛）|
| 赛道 | 05 SpaceTech & Satellite Intelligence（Bonus Track）|
| 团队 | `winnerwinner`（个人参赛）|
| 学校 | Jinling Institute of Technology ✅ 证书上写对了 |
| 验证链接 | `https://unstop.com/certificate-preview/444843e7-f3fb-4c9e-a087-39b62b607e2f` |

---

## 一、简历条目（中文版）

### 格式 A：单行式（空间紧张时用）

```
ORCATHON 2026 全球黑客马拉松（Dev Harbour 主办 · SpaceTech 赛道）· 参赛证书
独立开发并提交 TerraWatch 卫星地表变化检测系统，变化 IoU 0.905、人为/自然
变化区分准确率 88.1%（200 个留出场景实测）；GitHub 公开源码与线上演示
```

### 格式 B：带子条目（推荐，HR 和评审最爱看）

```
ORCATHON 2026 全球黑客马拉松 · SpaceTech 赛道 · 参赛证书 · 2026.10（单人参赛）
项目：TerraWatch — 卫星地表变化检测与自动简报系统（Python / PyTorch / NumPy）
· 独立完成端到端开发：自建合成遥感数据生成器（精确像素级 ground truth）、
  5,639 参数变化检测网络、纯 NumPy 部署推理、评测框架与线上 Web 应用
· 变化检测 IoU 0.905、像素准确率 99.3%、人为/自然变化区分准确率 88.1%，
  均在 200 个与训练集完全不重叠的留出场景上实测得出
· 定位并修复 3 处标签级数据缺陷（挖掘/洪水/城市扩张三类 IoU 0.03→0.57、
  0.00→0.64、0.07→0.61），通过消融实验证明问题源于数据可分性而非模型容量
· 线上演示 + 22 秒演示视频 + 开源全部源码（GitHub: ASylviaofmine/terrawatch）
```

### 格式 C：一行放简历「竞赛经历」表格

| 赛事 | 赛道/组别 | 时间 | 结果 | 关联产出 |
| --- | --- | --- | --- | --- |
| ORCATHON 2026（Dev Harbour 全球黑客马拉松，654 人参赛） | SpaceTech & Satellite Intelligence | 2026.10.25 | 参赛证书（有效提交） | TerraWatch：IoU 0.905，源码与线上演示公开 |

---

## 二、简历条目（英文版）

### Single line

```
ORCATHON 2026 Global Hackathon (Dev Harbour, SpaceTech & Satellite Intelligence track) — Participation Certificate
Solo-developed TerraWatch, a satellite surface-change detection system; change IoU 0.905 and 88.1% accuracy at separating human-made from natural change, measured on 200 held-out scenes; full source and a live demo are public on GitHub.
```

### With bullets

```
ORCATHON 2026 Global Hackathon — SpaceTech & Satellite Intelligence Track — Participation Certificate (Oct 2026, solo)
Project: TerraWatch — satellite surface-change detection with automated incident reporting (Python / PyTorch / NumPy)
· Built end to end and solo: a procedural remote-sensing data generator with exact pixel-level ground truth, a 5,639-parameter change-detection network, a NumPy-only inference path for deployment, an evaluation harness, and the deployed web app
· Change IoU 0.905, pixel accuracy 99.3%, and 88.1% accuracy at distinguishing human-made from natural change, all measured on 200 scenes generated from a seed range disjoint from training
· Diagnosed and fixed three label-level data defects by measurement (excavation, flooding and urban expansion each recovering from IoU 0.03/0.00/0.07 to 0.57/0.64/0.61), demonstrating the failures came from class separability rather than model capacity
· Shipped a live deployment, a 22-second demo video, and the complete source (github.com/ASylviaofmine/terrawatch)
```

### One-liner for a table

| Competition | Track | Date | Result | Linked output |
| --- | --- | --- | --- | --- |
| ORCATHON 2026 Global Hackathon (Dev Harbour, 654 participants) | SpaceTech & Satellite Intelligence | Oct 2026 | Participation Certificate (valid submission) | TerraWatch — IoU 0.905, public source and live demo |

---

## 三、LinkedIn 用（如果要在 LinkedIn 上加）

```
ORCATHON 2026 — SpaceTech & Satellite Intelligence Track · Participation Certificate
Oct 2026 · 4h 30m build window, solo

I built TerraWatch: give it two images of the same ground, months apart, and it
tells you what changed, what KIND of change it was, and whether a human or
nature caused it — then writes the incident report.

Measured on 200 held-out scenes: change IoU 0.905, 88.1% accuracy at
separating human-made from natural change.

The part I'm happiest with is what the numbers taught me. My first model
scored IoU 0.03 on excavation and 0.00 on flooding, and the obvious move was
to make the network bigger. The real cause was in my data: those textures were
identical to the bare soil and standing water already in the scene, so the
image pairs were ambiguous by construction. Fixing the labels took it to 0.57
and 0.64. The second time round, the same thing happened to urban expansion.

Live demo: https://terrawatch-change-detection.app.workbuddy.host/?sharecode=WSQ674JJXR
Code: https://github.com/ASylviaofmine/terrawatch
```

---

## 四、Erasmus+ / 联合培养申请表用的一页纸

> 申请表常有「附件：补充材料」或「其他成果」栏，把这页附在成绩单/获奖证明后面。

```
李海钰 · 2217011031 · 金陵科技学院 自动化专业

补充材料：ORCATHON 2026 全球黑客参赛项目（SpaceTech 赛道，单人参赛）

项目名称   TerraWatch — 卫星地表变化检测与自动简报系统
赛事       ORCA CODERS: ORCATHON 2026（Dev Harbour 主办，线上，654 人参赛）
赛道       05 SpaceTech & Satellite Intelligence
时间       2026 年 10 月 25 日（4.5 小时限时构建 + 部署 + 提交）
结果       有效提交，获颁参赛证书（Participation Certificate）

项目内容
  输入同一片区域相隔数月的两期遥感影像，输出：逐像素变化图、变化类型、
  人为/自然属性判定，以及一份可交付的人工复核简报。识别六类变化：
  森林砍伐、城市扩张、露天开挖、新建道路、洪涝、火灾迹地。

技术要点
  · 数据   自建程序化遥感场景生成器，输出精确像素级 ground truth，
           使每个评价指标都可测量而非估算
  · 模型   5,639 参数全卷积网络，膨胀卷积扩大感受野至 15×15
  · 部署   纯 NumPy 推理（服务端不依赖 PyTorch），自检脚本断言与 PyTorch
           模型误差 < 1e-4，确保线上跑的就是训练出的那个模型
  · 评测   200 个与训练集种子范围完全不重叠的留出场景

实测结果（200 个留出场景）
  变化检测 IoU                0.905
  像素准确率                  99.3%
  人为/自然变化区分准确率     88.1%
  各类 IoU：无变化 0.995 · 新建道路 0.944 · 火灾迹地 0.874 ·
            森林砍伐 0.753 · 洪涝 0.640 · 城市扩张 0.608 · 开挖 0.430

方法论收获（本项目的可迁移部分）
  项目迭代过程中出现三次「模型指标异常低」的情况。最终定位到的原因都不在
  网络，而在数据：当某一类的纹理与场景中既有地表高度相似时，两期影像在
  语义上就是歧义的，此时任何模型都无法仅凭两幅图区分。修正标签后，该类
  IoU 从 0.03 / 0.00 / 0.07 回升至 0.57 / 0.64 / 0.61。这一过程让我建立
  了「先怀疑标签、再怀疑模型」的工作习惯，也直接迁移到了我其他项目中。

公开材料
  源码与完整评测   https://github.com/ASylviaofmine/terrawatch
  线上演示         https://terrawatch-change-detection.app.workbuddy.host/?sharecode=WSQ674JJXR
  演示视频         https://youtu.be/gpFW010gmpE

声明
  本项目全部代码由本人独立完成；使用 AI 工具辅助了部分脚本起草与文档撰写，
  已在仓库 README「What I built vs. what was AI-assisted」一节完整披露。
```

---

## 五、证书文件拿到后放哪 / 怎么命名

**建议路径**（和你现有材料一致）：
```
C:\Users\Lee\Desktop\apply for\
  └── ORCATHON2026\
       ├── ORCATHON2026_参赛证书_HaiyuLiLi.jpg   ← 证书原图（已归档）
       ├── ORCATHON2026_参赛证书_HaiyuLiLi.pdf   ← 提交用（已转好）
       ├── ORCATHON2026_姓名说明.pdf            ← 若未重发，需手写签名后扫描
       ├── ORCATHON2026_TerraWatch_项目说明.pdf  ← 上面第四节的内容
       └── screenshot_unstop_submission.png     ← 提交成功的截图
```

> 📌 文件名保留了证书上的实际姓名 `HaiyuLiLi`——这样日后和官方核对
> 一目了然，不会误以为是另一份材料。**如果官方重发了正确姓名的版本，
> 把新文件加进来，并把旧的移到 `old/` 子目录。**

> ⚠️ `apply for` 目录里现有 11 个随机名 jpg
> （`0c15800947d0147f5b81bece3b859778.jpg` 这种），以后根本认不出哪个是
> 哪个证书。建议**只改名为可读的**，不要动已有的——那些 jpg 可能对应着
> 不同表格的具体要求，改名会打断对应关系。
>
> ⚠️ 改名/移动前先备份。这个目录里的东西是要交给学校和评审的，**不要
> 冒险操作**。我建议只**新增** ORCATHON2026 子目录，不动任何现有文件。
> （上面那个子目录已经建好并放入了证书，别的什么都没碰。）

---

## 六、还需要你确认的一件事

证书下载下来后，检查上面的**姓名**是否和报名时填的完全一致
（含大小写、拼写）。如果不一致，**先在 Unstop 后台改**，或联系主办方
（Dev Harbour 的 WhatsApp 群 / LinkedIn）要一份正确的。
