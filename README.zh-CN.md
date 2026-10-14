<div align="center">
  <img src="assets/logo.png" width="600" alt="OmniMed-Jev">
  <h1>OmniMed-Jev</h1>
  <p><b>把 LVLM 的置信度做成可信的医学多模态决策</b></p>
  <p>
    <a href="https://arxiv.org/abs/2610.00381"><img src="https://img.shields.io/badge/arXiv-2610.00381-b31b1b.svg" alt="arXiv"></a>
    <a href="https://lytang63.github.io/OmniMed-Jev/"><img src="https://img.shields.io/badge/Project_page-OmniMed--Jev-087f75.svg" alt="项目主页"></a>
    <a href="https://huggingface.co/xmutly/OmniMed-Jev-4B"><img src="https://img.shields.io/badge/Weights-HuggingFace-yellow.svg" alt="Weights"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python"></a>
    <a href="https://huggingface.co/google/medgemma-1.5-4b-it"><img src="https://img.shields.io/badge/backbone-MedGemma--1.5--4B-lightgrey.svg" alt="Backbone"></a>
  </p>
  <p>
    <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
  </p>
</div>

**项目主页：** [lytang63.github.io/OmniMed-Jev](https://lytang63.github.io/OmniMed-Jev/)

一个医学模型「答对了」和「它自报的置信度可信」是两件事。OmniMed-Jev 试图通过改变
**模型输出什么** 来弥合这个差距：不再把诊断、发现列表、计数或测量值序列化成文本，
而是对**推理时才给定的候选集**输出一个归一化的概率分布。

底座是 MedGemma-1.5-4B，外加一个共享的决策打分头。一次前向就能装下一个医学 state 的
全部决策——一张胸片上的 14 个发现、一张血涂片上的 4 个细胞计数——每一个都以
「说什么就是什么」的概率返回。

---

## ⭐ 核心亮点

| | 是什么 | 证据 |
|---|---|---|
| 🎯 | **校准是训练出来的，不是事后修的。** 决策分布本身就是模型的输出对象。 | 四个任务族里三个的 reliability error 降到 **1/9.1、1/44、1/126**——见[决策质量](#-结果) |
| ⚖️ | **没有精度税。** 点预测持平；论文不主张预测上的优势，也不需要。 | [点预测](#-结果)——4 个族里 3 个的差距小于配对区间的分辨力，counting 是唯一例外 |
| 🌐 | **Omni 覆盖，共用一个打分头。** 8 种模态标签、15 个源数据集、4 个任务族、3 种决策类型。 | [数据集](#-数据集) |
| 🔧 | **推理时可编辑的决策空间。** 打乱、截短、删掉正确答案——都不用重训。 | [探针实验](#-结果)——固定输出头物理上无法接受这些测试 |

唯一输给生成式基线的是 **counting**，而且给出了机制解释而非藏起来：把有序网格上的
分布折叠成期望，会把极端值往中间拉。

---

## 🔬 动机：token 概率不是决策概率

两个模型对同一张图给出同一个诊断，犯错频率也一样。一个在答错的样本上给出高置信度；
另一个在快要答错时降低置信度。它们的准确率完全相同——但只有第二个能用来决定哪些
病例需要复核。

通用医学多模态模型在「感知什么」和「解决什么」上确实已经通用了。这种通用性止步于
输出端。诊断、发现集合、细胞计数和连续测量值都是**结构不同**的有界决策，却全部被
序列化成文本，于是答案背后的计算就是模型写散文时用的那个条件 token 分布：

```
p_LM(y | S, Q) = Π_t p(y_t | y_<t, S, Q)
```

**一个字符串的概率，不等于这个字符串所编码的那个决策的概率。** 从前者恢复后者是一次
外挂的推理步骤——约束解码、显式候选打分、事后校准——它们处理的都是模型从未被训练去
产生的量。医学文献反复量到了后果：自报置信度高于准确率、跨专科校准不一致、以及
「不确定就转人工」的策略因为消费的置信度根本不是决策概率而行为不可预测。

<p align="center">
  <img src="assets/figures/fig1_overview.png" width="820" alt="图 1。从生成式接口到 System One 决策接口。">
</p>

*图 1。(a) 通用医学 MLLM 覆盖多种模态与任务，但所有输出都走同一条自回归文本通道。
(b) OmniMed-Jev 换掉的是输出通道，不是底座。(c) 主要结论。*

---

## 🧩 OmniMed-Jev：运行时候选集上的类型化决策

一个决策是三元组 `D = (Q, T, C)`：问题、类型、有限候选空间 `C = {c₁, …, c_K}`。
**候选集是模型输入的一部分**，不是固定输出头的索引，模型对它返回一个归一化向量：

```
p_θ(c | S, D_j),  c ∈ C_j,   Σ_{c ∈ C_j} p_θ(c | S, D_j) = 1
```

三种类型覆盖了结构化临床预测里的有界决策：

| 类型 | 候选空间 | 决策规则拿到什么 | 用于 |
|---|---|---|---|
| **Choice** | 无序有限集合 | 每个类的 `p(c)`，不只是赢家的名字 | 器官/组织类的鉴别诊断 |
| **Noul** | `{No, Yes}`，每个 finding 一个 | 每个 finding 各自一个概率，规则可以逐个升级处理 | 胸片上的发现 |
| **Score** | 有序网格 `{v₁, …, v_K}` | 网格上的分布 → 期望与区间 | 细胞计数、射血分数、视网膜分级 |

`Score` 的候选**从不打乱**：位置就是序数索引，且候选文本是自足的（`- 45 ⟩` 已经说了
45），所以挪动一个候选只改变它占的槽位，永远不改变它怎么读。

第四种类型 `Span`（把检测框表示成 span 决策）在接口契约里有定义，但本文不评测：
坐标头塌成了常数框，没有定位住发现，所以论文对检测不做任何方向的声明。

### 与 Jev 的关系

OmniMed-Jev 是 decision-interface 思想在医学上的**实现与检验**，不是某个通用决策模型的
医学移植版：它保留医学多模态底座，加上一个共享的候选打分机制。它承载的 Jev 性质：

| Jev 元素 | 生成式 SFT | OmniMed-Jev |
|---|---|---|
| **动态决策空间**——候选是输入语义，不是固定分类头的 index | ✗ 固定词表 | ✅ 候选文本进入上下文；打分头读的正是它在打分的那一项 |
| **完整分布**——归一化分布，不是字符串或 argmax | ✗ 下一 token 似然 | ✅ 候选集上的 `softmax` |
| **类型化决策**——Choice / Noul / Score 是三种不同的原语 | ✗ 一律文本 | ✅ 三者共用一条交叉熵 |
| **一个 state → 多个决策**——单次前向 | 一个 prompt 一个答案 | ✅ 一张胸片 14 个 `Noul`，一张涂片 4 个计数 |
| Workflow 组合——窄决策由确定性代码串起来 | ✗ | ✅ presence → count → localize |

因为打分头读的是**被标记位置**的隐状态，不需要投影到 26 万词的词表：

```
z_jk = f_φ(h_state, h_{c_jk}),      p_jk = exp(z_jk) / Σ_l exp(z_jl)
```

三种类型共用**一个**目标函数——模型分布与标注导出的目标分布之间的交叉熵：

```
L(θ) = − (1/M) Σ_j Σ_k q_jk log p_jk
```

`Noul` 取 `C = {False, True}` 即归约到同一条式子；`Score` 配插值软目标走的也是同一条
路径，所以 one-hot 标签和软标签是同一份代码。

---

## 📊 结果

下面每个数字都来自一个**接口受控**的对照：两个臂共享底座、训练图像、预处理、adapter
配置与步数预算，差别只在输出目标。两者都在 step 2424 上评测，测试集是 726 条
held-out state，两侧同一批。

### 点预测——不为任何一方主张优势

| 任务族 | 指标 | Gen-SFT | OmniMed-Jev | Δ |
|---|---|---:|---:|---:|
| Classification | Balanced accuracy ↑ | 0.6469 | **0.6892** | +0.0422 |
| Multi-label | F1 ↑ | 0.5449 | **0.5618** | +0.0170 |
| | Precision ↑ | 0.5679 | **0.5950** | +0.0271 |
| | Recall ↑ | 0.5498 | **0.5602** | +0.0104 |
| Counting | MAE ↓ | **2.1429** | 2.5159 | +0.3730 |
| Regression | MAE ↓ | 3.3400 | **2.9488** | −0.3912 |

分类、多标签与回归的差距在 95% 下**无法与零区分**（配对 bootstrap 区间分别为
`[−0.0196, +0.1059]`、`[−0.0264, +0.0620]`、`[−0.8756, +0.0173]`），多标签的排序在
pooled micro 平均下会反转，而且这个对比随 checkpoint 变化。唯一能与零区分的任务级
差异是 counting 的 MAE——而它有利于生成式基线。

### 决策质量——接口真正改变模型的地方

每一格是 **Gen-SFT / OmniMed-Jev**；ECE、reliability、AURC 越低越好，resolution 越高越好。

| 任务族 (n) | ECE ↓ | Reliability ↓ | Resolution ↑ | AURC ↓ |
|---|---:|---:|---:|---:|
| Classification (200) | 0.2589 / **0.0806** | 0.1002 / **0.0110** | **0.0938** / 0.0860 | 0.2020 / **0.1351** |
| Multi-label (2,795) | 0.2981 / **0.0114** | 0.1294 / **0.0010** | **0.0776** / 0.0102 | 0.2474 / **0.0126** |
| Counting (201 / 126) | 0.4204 / **0.0408** | 0.1824 / **0.0041** | 0.0000 / **0.1087** | 1.0000 / **0.3890** |
| Regression (200) | 0.1267 / **0.0913** | 0.0258 / **0.0205** | **0.1426** / 0.1268 | **0.2272** / 0.2280 |

> **基线的概率是怎么来的。** 生成式模型只输出答案字符串，不输出分布，所以它这一栏的数字
> 是**重建**出来的：每个候选的续写在语言模型下按对数概率求和，再在候选集上归一化。它的
> 训练从未让这些似然之和等于 1——这是一个伪分布，而 OmniMed-Jev 的分布就是训练目标本身。
> 有一格完全由这个差异决定：counting 的 AURC 1.0000 与 resolution 0.0000 是重建的机械
> 后果，因为在 33 点整数网格上它排名第一的候选从未等于标注计数（126 行里 0 行命中），而
> 一个 top-1 从不正确的分布无法区分难易样本。这一格应当读作「关于重建的结论」，而不是
> 「生成式模型一般携带多少置信度信息」。

<p align="center">
  <img src="assets/figures/fig3_reliability_resolution.png" width="600" alt="图 3。可靠性–分辨率平面上的分界。">
</p>

*图 3。横轴 reliability error，纵轴 resolution。四个 OmniMed-Jev 族全部落在
`resolution = reliability` 分界的绿色一侧；生成式基线在 classification、multi-label
和 counting 上越到了另一侧。*

**这个分离才是重点。** 生成式基线通过**判别力**测试的次数多于通过**校准**测试的次数：
在四个族里的三个，它的 reliability error 高于 resolution。它能有效地给样本排序，但报出
的数字高估了它答对的频率。OmniMed-Jev 把这个模式反了过来——reliability error 降到
1/9.1（分类）、1/44（计数）、1/126（多标签），而 resolution 在分类与回归上保持可比
——于是同一套排序被表达成了一个下游规则可以消费的量。

每个指标的定义与适用范围见 [`docs/METRICS.md`](docs/METRICS.md)。

这也解释了为什么论文**同时**报 reliability 与 resolution，绝不只报 ECE：一个永远输出
类别先验的模型 reliability error 接近零、resolution 恰好为零，只看 ECE 无法区分这两类
失败模式。

### 选择性预测

<p align="center">
  <img src="assets/figures/fig4_selective_prediction.png" width="820" alt="图 4。按覆盖率报告的选择性预测。">
</p>

*图 4。只回答最自信的 50/80/90/95/100% 决策时的 top-label 错误率。AURC 为
OmniMed-Jev / Gen-SFT。*

按决策臂的概率排序、只保留最自信的一半，错误率远低于在相同覆盖率下按基线重建置信度
排序，而且差距随覆盖率扩大。多标签识别的选择性风险曲线下面积从 0.2474 降到
**0.0126**——在任何覆盖率水平上，转人工策略承担的错误都降到 1/20。

### 训练动态

<p align="center">
  <img src="assets/figures/fig2_dynamics.png" width="820" alt="图 2。四个任务族的学习曲线。">
</p>

*图 2。两个臂在共享的 726 条测试集上的全部十个配对 checkpoint。虚线标出对照点 step 2424。*

两个接口不仅在终点不同，轨迹也不同：生成式基线在分类上前期领先，在 step 1616 之后被
反超；回归上它经过一次 OmniMed-Jev 没有的中途大幅波动。counting 是生成式基线全程领先的
唯一一个族。

### 推理时可编辑的决策空间

因为候选语义是**输入**，决策空间可以在推理时改动而不重训。这些操作在生成式接口下没有
对应物（除非另加一套候选打分机制），在固定分类头下则**根本无法定义**——它的 logits 按
位置索引。

| 探针 | 检验 | @1616 | @2424 |
|---|---|---:|---:|
| 打乱候选顺序，total variation ↓ | 动态决策空间 | 0.1284 | 0.1413 |
| 打乱候选顺序，top-1 一致率 ↑ | | 0.6589 | 0.6622 |
| └ 分类 / 计数 / 回归 | | 0.91 / 0.37 / 0.70 | 0.90 / 0.42 / 0.67 |
| 候选缩到 2 个时准确率 ↑ *（随机基线 0.50）* | | 0.8735 | 0.8719 |
| 候选缩到 4 个时准确率 ↑ *（随机基线 0.25）* | | 0.6717 | 0.7322 |
| 删掉正确答案后改选原次优 ↑ | 完整分布 | 0.8885 | 0.8752 |

候选缩到两个时准确率仍远高于随机；删掉正确答案后模型会改选一个合理的次优，而不是崩掉。

### counting 这条边界

counting 是生成式基线赢的那个族，原因是 `Score` 表示法的具体设计，而不是方法本身。计数
是一个有界测量值，标注是单个整数；本文把它表示成有序网格上的分布再折叠成期望，而基线
直接吐出那个整数。最可能的机制是折叠本身：大范围网格上的期望会把极端值往中间拉，于是
在一个较宽的高值区间上不确定的模型会报一个更靠近中心的值。这是全套任务里概率对象唯一
让精度变差的地方，而这个机制足够具体，是**可以修**的，不是原理性的。

### 主张的边界

这个对照无法把「接口」与它带来的「更密的监督」（同样 23,558 张图，按决策语义分解后变成
**155,729** 个决策级 target）和「额外训练了一个决策头」分开。所以测到的是**按此实现**的
decision modeling，而不是接口本身。概率是在来自训练同源的 held-out 数据上回顾性评测的；
per-class、per-source 和分布外行为都没有测。两个臂所有不对等的地方逐条列在
[`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

---

## 🗂️ 数据集

两个臂训练在同一批 **23,558 个 state**、评测在同一批 **726 个 held-out state** 上。整套
覆盖 **15 个源数据集**、**8 种模态标签**、**4 个任务族**；划分是**我们自己构建的**，不是
沿用某个公开 benchmark，所以两个臂都不能从「对某个被广泛使用的测试集更熟悉」中获益。

| 任务族 | 类型 | 源数据集 | 模态 | States | Decisions | Test | 下载 |
|---|---|---|---:|---:|---:|---:|---|
| **Classification** | Choice | TissueMNIST, PathMNIST, OCTMNIST, Organ{A,C,S}MNIST, BloodMNIST, DermaMNIST, PneumoniaMNIST, BreastMNIST | 组织学、病理、CT、血液学、皮肤科、眼科 | 10,000 | 10,000 | 200 | [MedMNIST+ 224](https://zenodo.org/records/10519652) |
| **Multi-label** | Noul ×14 / ×13 | ChestMNIST, [ChestX-Det](https://github.com/Deepwise-AILab/ChestX-Det-Dataset) | 胸片 | 10,000 | 139,537 | 200 | [MedMNIST+ 224](https://zenodo.org/records/10519652) |
| **Counting** | Score K=33 | [TXL-PBC](https://huggingface.co/datasets/KeenForgeAI/TXL-PBC-corrected) | 血液学显微 | 878 | 3,512 | 126 | `huggingface-cli download KeenForgeAI/TXL-PBC-corrected --repo-type dataset` |
| **Regression** | Score K=101 / K=5 | [CAMUS](https://huggingface.co/datasets/MedOtter/CAMUS), RetinaMNIST | 超声、眼底 | 2,680 | 2,680 | 200 | [MedOtter/CAMUS](https://huggingface.co/datasets/MedOtter/CAMUS) |
| **合计** | 3 种 | **15 个数据集** | **8 种** | **23,558** | **155,729** | **726** | |

分类把十个 MedMNIST 集合合并成一个在器官/组织类上做选择的 Choice 问题。回归混合了两种
网格——射血分数 101 点、视网膜分级 5 点——由同一个 `Score` 机制承载，这正是它们可比的原因。

### 划分是怎么构建的

```
源数据集 ──► 转换树 ──► 预处理行 ──► 均衡子集 ──► 决策 state
 (15 个)     datas/converted/   529,673 train    29,842 train    23,558 state
             omnimed-suite/      121,638 test      (按任务均衡)    155,729 decisions
```

- **Train** —— 任务均衡子集：每个任务族最多 10,000 行，空答案上限 25%，再限制到四个决策
  族。即 29,842 − 3,614 − 2,670 = **23,558** 行，一行一张图。
- **Validation** —— **600** 个 state，每个源按种子抽 40 个。这是选对照 checkpoint 用的 dev 集。
- **Test** —— **726** 个 state，held out，两个臂完全相同。

划分本身随仓库发布——uid 清单加 test/validation 的 state 文件——所以不重训也能复现论文的
数字。见 **[`data/README.md`](data/README.md)**。

---

## 🚀 快速开始

```bash
git clone https://github.com/lytang63/OmniMed-Jev && cd OmniMed-Jev
conda create -n omnimedjev python=3.12 -y && conda activate omnimedjev
export CONDA_ENV=omnimedjev          # 让 scripts/env.sh 自动激活
pip install -e .
```

### 1. 权重与数据

下载脚本需要 `aria2c` 和 `huggingface-cli`，`pip install -e .` 不会装上。

```bash
./scripts/download_models.sh                    # MedGemma-1.5-4B -> models/
./scripts/download_datasets.sh                  # -> datas/raw/（7 组）
python scripts/convert_datasets.py medmnist chestx camus txlpbc
./scripts/run_preprocess.sh start               # 异步运行，用 `status` 查看进度

# 按任务均衡抽样，再保留四个决策族（29,842 -> 23,558）
python scripts/make_train_subset.py \
    --source datas/preprocessed/Preprocessed-omnimed-suite/train.jsonl \
    --out datas/balanced/train.jsonl --per-task 10000 --empty-cap 0.25
python scripts/make_train_subset.py \
    --source datas/preprocessed/Preprocessed-omnimed-suite/validation.jsonl \
    --out datas/sft/Preprocessed-omnimed-suite/validation.jsonl --per-task 0
python scripts/make_eval_subset.py --out datas/eval

# 两个臂共用评分的决策 state
python scripts/build_medjev_data.py \
    --src datas/sft/Preprocessed-omnimed-suite/train.jsonl --out datas/medjev/train.jsonl
python scripts/build_medjev_data.py \
    --src datas/preprocessed/Preprocessed-omnimed-suite/validation.jsonl \
    --out datas/medjev/validation.jsonl --per-source 40
python scripts/build_medjev_data.py \
    --src datas/eval/Preprocessed-omnimed-suite/testing.jsonl --out datas/medjev/testing.jsonl
```

压缩包解压、把 29,842 行过滤成 23,558 行的 uid 过滤步骤、以及链条上每一条路径，都在
[`docs/REPRODUCE.md`](docs/REPRODUCE.md)。如果不想重建任何东西，划分本身随仓库发布，
见 [`data/splits/`](data/README.md)。

### 2. 训练

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_train_medjev.sh   # OmniMed-Jev（两阶段）
CUDA_VISIBLE_DEVICES=0 ./scripts/run_train_sft.sh      # 生成式基线
```

每臂一张 A800-80GB，各约 13 小时。逐项的完整配置、以及两个臂唯一那一处差异，见
[`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

### 3. 评测

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_full_eval.sh
```

给两个臂的每个 checkpoint 打分、跑决策空间探针、从 logits 恢复生成式臂的分布。每个阶段
都会跳过已有输出的部分，可以随时重跑。

### 4. 或者直接用已发布的权重

已发布的 checkpoint 是 `google/medgemma-1.5-4b-it` 上的约 150 MB adapter + 决策头。把推理
指向一个 state 文件，它会写出折叠后的预测以及完整的决策分布：

```bash
python scripts/infer_medjev.py \
    --checkpoint output/medjev-lora \
    --data data/splits/testing_states.jsonl \
    --out-dir datas/eval/checkpoint-final
```

---

## 📦 随仓库发布的划分文件

| 文件 | 行数 | 内容 |
|---|---:|---|
| [`data/splits/training_uids.txt`](data/splits/training_uids.txt) | 23,558 | 训练 uid |
| [`data/splits/validation_uids.txt`](data/splits/validation_uids.txt) | 600 | dev uid |
| [`data/splits/testing_uids.txt`](data/splits/testing_uids.txt) | 726 | 测试 uid |
| [`data/splits/testing_states.jsonl`](data/splits/testing_states.jsonl) | 726 | 测试**决策 state**（含 target） |
| [`data/splits/validation_states.jsonl`](data/splits/validation_states.jsonl) | 600 | 验证决策 state |
| [`data/splits/testing_sft.jsonl`](data/splits/testing_sft.jsonl) | 726 | 同一批测试行，生成式臂的格式 |
| [`data/splits/chestx-det_validation_manifest.json`](data/splits/chestx-det_validation_manifest.json) | 289 | ChestX-Det 的留出规则与文件清单 |

state 里的图片路径**相对于仓库根目录**，所以只要把转换树落在
`datas/converted/omnimed-suite/`，它们可以直接加载。两个臂按 `uid` 对齐：

```
<源划分>:<源数据集>:<问题文件>:<case>:<task_type>:<case>
```

**权重和图片不进仓库**；清单加公开源数据集足以把它们精确重建出来。

---

## 📁 仓库结构

```
OmniMed-Jev/
├── omnimedjev/                  包
│   ├── engine/
│   │   ├── medjev.py            ⭐ 唯一契约：marker 编码、决策空间、
│   │   │                          共享打分头、统一损失、折叠规则
│   │   ├── medjev_model.py      模型包装 + 批次拼装
│   │   ├── medjev_metrics.py    校准 / 选择性预测指标
│   │   ├── preprocess.py        转换后的 question JSON -> SFT 行
│   │   ├── train.py             生成式臂训练
│   │   ├── infer.py             生成式臂推理
│   │   ├── evaluate.py          基于字符串的任务评测
│   │   └── check_dataset.py     数据集检查
│   ├── interfaces.py            各阶段入口 + 环境检查
│   ├── __entry__.py             `omnimedjev` 命令行
│   └── vars.py                  实验配置
├── scripts/
│   ├── convert_datasets.py      源数据集 -> 转换树
│   ├── download_datasets.sh     下载源数据集
│   ├── download_models.sh       从 ModelScope 下载 MedGemma
│   ├── build_medjev_data.py     行 -> 决策 state
│   ├── train_medjev.py          两阶段训练（先头，再头 + LoRA）
│   ├── infer_medjev.py          推理 + workflow + 折叠
│   ├── probe_medjev.py          permute / restrict / drop-gold / add-none
│   ├── eval_medjev_metrics.py   概率指标 -> metrics.json / .md
│   ├── eval_sft_logit_conf.py   从 logits 读生成式臂的分布
│   ├── sweep_medjev.py          扫全部 checkpoint：推理 -> 折叠 -> 评测 -> 指标
│   ├── sweep_sft.py             同上，生成式臂版
│   ├── compare_medjev.py        并排对照表
│   ├── run_train_medjev.sh      ⭐ 启动已发布的 run
│   ├── run_train_sft.sh         ⭐ 启动基线
│   ├── run_full_eval.sh         ⭐ 全量评测 + 探针
│   └── env.sh                   启动脚本用的缓存/代理/conda 设置
├── configs/                     已发布 run 的配置
│   ├── train-sft.yaml           生成式基线
│   ├── preprocess.yaml          手工跑六族预处理时用
│   ├── evaluate.yaml            任务评测器
│   ├── evaluate-medjev.yaml     手工评测决策臂
│   └── infer-*-conf*.yaml       失败的自报置信度尝试（附录 F）
├── data/                        发布的划分（+ 图的数字快照）
├── docs/
│   ├── REPRODUCE.md             端到端复现、硬件、统计协议
│   └── METRICS.md               每个指标的定义与适用范围
└── assets/                      徽标、论文配图
```

> 代码与脚本里把这个模型叫 **MedJev**——开发期间的内部名字。它和论文里的 OmniMed-Jev
> 是同一个模型。

### 权重

两个已发布的 checkpoint 都在 HuggingFace：**[`xmutly/OmniMed-Jev-4B`](https://huggingface.co/xmutly/OmniMed-Jev-4B)**
——决策臂在仓库根目录，生成式基线在 `generative-baseline/`。

| 产物 | 权重 | 说明 |
|---|---:|---|
| `medjev-lora` | ~150 MB | LoRA adapter + 决策头；单张 A800 约 13.95 小时 |
| `sft-lora` | ~75 MB | 只有 LoRA adapter；单张 A800 约 12.86 小时 |

完整 checkpoint 目录比权重本身大（约 230 MB / 180 MB），因为训练 checkpoint 还带
optimizer 状态，推理用不到。

---

## ⚖️ 限制与伦理

- **只覆盖有界决策。** 接口面向候选集可枚举的决策。自由文本报告生成与坐标回归不在本文
  评测范围内；检测的实现也没有产出可用的定位结果，所以论文对该族不做任何方向的声明。
- **监督被分解了。** 两个臂看到同样的图像，但按决策语义分解标注后，决策臂拿到 155,729 个
  target，而基线只有 23,558 个。更密的监督很可能解释了分类在后期的反超；这是一个假设，
  能检验它的消融实验还没跑。
- **评测范围。** 4 个任务族、726 条 held-out state、3 种决策类型，足以检验接口效应是否是
  普遍的，不足以确立临床效用。
- **未经临床验证。** 这些是回顾性预测器。本工作的任何部分都不是为诊断用途设计或验证的，
  决策概率没有在临床工作流中前瞻性评估过，而且一个能把决策**排序**好的模型并不因此就
  可以部署。转人工是把错误搬家，不是消除错误。
- **数据。** 本工作只使用已去标识的公开医学影像数据集，不收集任何新的患者数据。划分继承
  各自源数据集的许可条款。

---

## 📝 引用

```bibtex
@article{tang2026omnimedjev,
  title   = {OmniMed-Jev: Calibrating LVLM Confidence for Trustworthy Medical
             Multimodal Decisions via System One},
  author  = {Tang, Luyao and Chen, Cheng},
  journal = {arXiv preprint arXiv:2610.00381},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.00381}
}
```

另见 [`CITATION.cff`](CITATION.cff)。

## 🙏 致谢

基于 [MedGemma-1.5-4B](https://huggingface.co/google/medgemma-1.5-4b-it)，并在开发过程中
对照 [NanoJev](https://github.com/TianyuCodings/NanoJev)——[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
的开源 nano 复刻。评测集由公开数据集组装而成——MedMNIST+、ChestX-Det、TXL-PBC、CAMUS——
感谢它们的作者公开这些数据。

## 📄 许可

Apache-2.0 — 见 [LICENSE](LICENSE)。
