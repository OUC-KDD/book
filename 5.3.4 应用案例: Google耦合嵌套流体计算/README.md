# 🌊 STPDE-NET: 基于时空偏微分方程的物理信息神经网络海温预测项目

本项目提出了一种物理信息神经网络框架 **STPDE-NET**，通过将海洋热传输方程（Heat Transport Equation）嵌入到深度学习模型（CNN, ConvLSTM, ViT）中，实现对海表温度（SST）的高精度预测。

与纯数据驱动模型不同，STPDE-NET 在损失函数中显式引入了物理约束（如平流、扩散、热通量等项），从而提高了模型的物理一致性和泛化能力。

---

## 📁 项目结构

```
STPDE_NET-main/
│
├── code/                     # 模型代码主目录
│   ├── cnn/                  # 基于 CNN 的 STPDE-NET 实现
│   │   ├── STPDE-NET_train_mode.py   # 训练脚本 (包含物理损失计算)
│   │   ├── STPDE-NET_test_mode.py    # 测试脚本
│   │   ├── cnn_model.py              # CNN 网络架构定义
│   │   └── input_data.py             # 数据加载与预处理
│   ├── convlstm/             # 基于 ConvLSTM 的 STPDE-NET 实现
│   ├── vit/                  # 基于 Vision Transformer 的 STPDE-NET 实现
│   └── finite difference/    # 有限差分法 (数值模拟基准)
│
├── data/                     # 数据文件目录 (.npz 格式)
│   ├── coordinate_variables.npz  # 经纬度坐标网格
│   ├── deep_variables.npz        # 深层海洋变量 (Td, ud, vd)
│   ├── label.npz                 # 标签数据 (SST, MLD)
│   ├── other_variables.npz       # 其他物理变量 (通量等)
│   └── solar_radiation.npz       # 太阳辐射数据
│
└── README.md                 # 项目说明文档
```

---

## 🧩 功能概述

本项目主要包含以下核心功能模块：
1.  **数据处理 (`input_data.py`)**：读取多源 `.npz` 数据，构建包含物理场（温度、流场、辐射等）的训练数据集。
2.  **模型构建**：实现了多种骨干网络（CNN, ConvLSTM, ViT）与物理方程的融合。
3.  **物理约束训练 (`*_train_mode.py`)**：利用 `torch.autograd` 自动微分计算空间梯度，结合热收支方程计算物理残差损失。
4.  **对比评估**：提供了 STPDE-NET 与 传统 PINN (Physics-Informed Neural Networks) 及 纯数据驱动 (Data-driven) 方法的对比。

---

## 📊 一、数据处理

### 1. 数据来源
项目使用预处理好的 `.npz` 文件，涵盖了海洋热收支方程所需的关键变量：
- **状态变量**：海表温度 (SST), 混合层深度 (MLD)
- **动力变量**：海流速度 (u, v), 深层流速 (u_d, v_d)
- **热力变量**：净热通量 (Qnet), 短波/长波辐射, 深层温度 (T_d)

### 2. 预处理流程
- **加载**：`input_data.py` 负责加载所有物理变量，并将其重塑为 `(Time, Channel, Lat, Lon)` 的张量格式。
- **网格构建**：生成对应的经纬度网格 (xx, yy)，用于后续计算空间偏导数 ($\partial T/\partial x, \partial T/\partial y$)。
- **数据集划分**：
  - 训练集：前 2432 个时间步
  - 验证集：2432 ~ 2560 时间步
  - 测试集：2560 ~ 3646 时间步

---

## 🧠 二、模型原理 (STPDE-NET)

STPDE-NET 的核心创新在于将**海洋混合层热收支方程**作为先验知识嵌入模型。

### 物理方程
模型试图最小化预测值与以下物理方程的偏差：

$$ \frac{\partial T}{\partial t} \approx \underbrace{\frac{Q_{net}}{\rho C_p h}}_{\text{海气热通量}} - \underbrace{(u \cdot \nabla T)}_{\text{水平平流}} - \underbrace{w_e \frac{T - T_d}{h}}_{\text{垂直夹卷}} + \dots $$

### 实现细节
在 `STPDE-NET_train_mode.py` 中：
1.  **网络预测**：模型输入历史状态，输出预测的 SST 变化。
2.  **自动微分**：使用 PyTorch 的 `autograd` 计算输出 SST 对空间坐标 (x, y) 的梯度 `dT_x`, `dT_y`。
3.  **物理损失**：将预测的梯度代入上述方程，计算物理项推导出的温度变化量，并与网络直接预测的温度变化量进行约束。

---

## 💻 三、运行说明

### 1. 环境依赖
推荐使用 Anaconda 创建环境：
- Python >= 3.8
- PyTorch (支持 CUDA)
- NumPy
- Scikit-learn
- tqdm

### 2. 运行步骤 (以 CNN 模型为例)

**步骤 1：训练模型**
```bash
cd code/cnn
python STPDE-NET_train_mode.py
```
*程序将自动加载数据，开始训练，并保存模型权重文件 (`.pth`)。*

**步骤 2：测试模型**
```bash
cd code/cnn
python STPDE-NET_test_mode.py
```
*程序将加载训练好的权重，在测试集上进行预测，并输出 RMSE 误差指标。*

*(ConvLSTM 和 ViT 模型的运行方式类似，请进入对应的子文件夹操作。)*

---

## 🧾 四、实验结果与分析

通过引入物理方程，STPDE-NET 旨在解决以下问题：
1.  **物理一致性**：保证预测的温度场符合流体力学和热力学基本定律。
2.  **泛化能力**：在训练数据稀缺或分布外推的情况下，物理约束能提供额外的指导信息，防止过拟合。

实验结果通常使用 **RMSE (均方根误差)** 作为主要评价指标。对比纯数据驱动方法，STPDE-NET 在保持高精度的同时，生成的温度场在物理上更加合理。

---

## 📚 五、引用

Yuan, T., Zhu, J., Wang, W., Lu, J., Wang, X., Li, X., & Ren, K. 2023. A Space-Time Partial Differential Equation Based Physics-Guided Neural Network for Sea Surface Temperature Prediction. Remote Sensing, 15(14), 3498. DOI: https://doi.org/10.3390/rs15143498