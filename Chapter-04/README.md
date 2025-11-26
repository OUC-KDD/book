# CAMixer SAR 图像变化检测项目

本项目复现了 **CAMixer (Convolution and Attention Mixer)** 模型，专门用于合成孔径雷达（SAR）图像的变化检测任务。
该模型通过结合卷积（Convolution）与自注意力机制（Self-Attention），有效解决了 SAR 图像中斑点噪声干扰和全局特征提取不足的问题。

---

## 📂 项目结构

```text
CAMixer_Project/
├── 📂 data/                 # 存放原始 SAR 图像与真值图
│   ├── im1.bmp              # T1 时刻图像
│   ├── im2.bmp              # T2 时刻图像
│   └── gt.bmp               # 地面真值 (Ground Truth)
├── 📂 src/                  # 核心源码
│   ├── model.py             # CAMixer 网络模型定义 (PCAM, GFFN)
│   ├── dataset.py           # 数据集构建与 Patch 切片
│   ├── preclassify.py       # 预分类算法 (SRAD, FCM)
│   └── utils.py             # 评估指标与工具函数
├── 📂 outputs/              # 输出结果
│   └── result.png           # 最终生成的二值变化图
├── main.py                  # 主程序入口 (包含训练与推理全流程)
└── requirements.txt         # 环境依赖文件
```

## 📝 功能概述

本项目主要流程包含以下三个核心部分：

1. **预分类 (Pre-classification)**：利用传统无监督算法生成伪标签（Pseudo-labels）。
2. **模型训练 (Model Training)**：基于伪标签训练 CAMixer 网络，学习鲁棒的特征表示 。
3. **推理与评估 (Inference & Evaluation)**：对图像中的不确定区域进行推理，生成最终变化图并计算精度指标。

------

## 📊 一、数据处理与预分类 (preclassify.py)

### 1. 数据输入

输入为同一地理区域在不同时间（$t_1$ 和 $t_2$）拍摄的两幅 SAR 图像 。

### 2. 预分类流程

由于缺乏大量标注数据，本项目采用“现用现训”（Train-on-the-fly）的策略，首先通过 `src/preclassify.py` 生成训练所需的伪标签：

- **差异图生成**：使用对数比算子（Log-ratio operator）计算两幅图像的差异图 。
- **分层聚类**：使用分层模糊 C 均值聚类（Hierarchical FCM）将像素划分为三类 
  - **不变类 (Unchanged)**：标签为 0。
  - **变化类 (Changed)**：标签为 1。
  - **中间类 (Intermediate)**：难以判断的像素（标签设为 1.5），这部分将由深度学习模型进一步判断。

### 3. 数据切片 (Patch Generation)

- **输入形状**：将图像切分为 $9 \times 9$ 的 Patch 作为网络输入 。
- **训练集构建**：仅选取预分类中“高置信度”（即确定的变化或不变）像素及其邻域构建训练集 。

------

## 🧠 二、模型结构 (CAMixer)

CAMixer 的核心在于并行结合了局部卷积特征与全局注意力特征，主要由以下两个模块组成 ：

### 1. 并行卷积与注意力模块 (PCAM)

PCAM 模块通过并行路径同时提取局部与全局信息：

- Shift Convolution (移位卷积)：通过空间移位操作提取局部特征，计算效率高，公式如下 ：

  $$ \hat{X} = W_{1\times1}^2(shift(W_{1\times1}^1(X))) $$

- **Self-Attention (自注意力)**：捕获长距离依赖关系，增强全局语义理解 。

### 2. 门控前馈网络 (GFFN)

传统的 FFN 难以应对 SAR 图像的斑点噪声，本项目引入了 GFFN：

- **门控机制**：通过两个并行线性层的逐元素相乘（Element-wise multiplication）来选择性地强调重要特征 。
- **作用**：增强非线性特征变换能力，有效抑制斑点噪声的干扰 。

------

## ⚙️ 三、训练与验证

- **训练模式**：无监督/弱监督模式。针对每一组待检测图像，现场利用伪标签进行训练。
- **损失函数**：交叉熵损失 (CrossEntropy Loss)。
- **优化器**：Adam 优化器。
- **归一化策略**：对输入的每一个 Patch 进行独立的 Z-Score 归一化（减均值除标准差）。

------

## 📈 四、测试与结果

### 1. 测试方法

在推理阶段，模型主要针对预分类中的**中间类（不确定像素）**进行预测，或对全图像素进行精细化分类，最终合并生成完整的二值变化图。

### 2. 评估指标

项目输出以下指标与地面真值 (Ground Truth) 进行对比 ：

- **FP (False Positives)**：虚警像素数。
- **FN (False Negatives)**：漏警像素数。
- **OE (Overall Error)**：总错误数。
- **PCC (Percentage of Correct Classification)**：总体分类精度。
- **Kappa Coefficient**：Kappa 系数（评估一致性的关键指标）。

### 3. 实验结论

根据论文实验结果，在 Yellow River 和 Chao Lake 数据集上，CAMixer 相比传统方法（如 PCA-KM）和纯 CNN 方法（如 DDNet）均取得了更高的精度，且产生的变化图更加纯净，误报率更低。

------

## 🚀 五、运行说明

### 1. 生成数据与训练

使用 `main.py` 脚本即可一键完成预分类、数据构建、模型训练及推理。

Bash

```
# 基本运行命令
python main.py --im1 data/im1.bmp --im2 data/im2.bmp --gt data/gt.bmp

# 指定参数运行
python main.py --im1 data/im1.bmp --im2 data/im2.bmp --gt data/gt.bmp --epochs 20 --patch_size 9
```

### 2. 输出结果

运行结束后，程序将在控制台输出 PCC 和 Kappa 系数，并在 `outputs/` 目录下生成 `result.png`。

------

## 📦 六、环境依赖

推荐环境：

- Python $\ge 3.8$
- PyTorch $\ge 1.8$ (推荐使用 GPU 版本以加速推理)

依赖安装：

```bash
pip install numpy torch scipy scikit-image matplotlib einops tqdm
```

------

## 📖 七、引用

**Paper**: Haopeng Zhang, et al. "Convolution and Attention Mixer for Synthetic Aperture Radar Image Change Detection," IEEE Geoscience and Remote Sensing Letters, 2023.

**Code**: https://github.com/summitgao/CAMixer