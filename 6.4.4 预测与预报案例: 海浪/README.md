# 🌊 EarthFormer 全球有效波高（SWH）预测模型
*Physics-Guided Deep Learning for Global Wind–Wave Modeling*

本项目基于 **EarthFormer（Cuboid Transformer）架构**，利用 **ERA5 再分析风场数据（10 m 风速）**，实现全球海洋有效波高（Significant Wave Height, SWH）的高分辨率预测。

模型设计参考：

> **Wang, Xinxin & Jiang, Haoyu (2024).**  
> *Physics-guided deep learning for skillful wind-wave modeling.*  
> **Science Advances, 10(49), eadr3559.**

本仓库代码实现了该思想在 **全球 0.5° 分辨率海浪预测任务** 中的完整训练、验证与 **Epoch Ensemble 推理评估流程**。

---

## 📁 一、项目结构

```text
global_ocean/
│
├── earthformer/                     # EarthFormer 源码
│   └── cuboid_transformer/
│
├── model/                           # 模型 checkpoint
│   └── latest_checkpoint.pt
│
├── logs/                            # TensorBoard 日志
│
├── mask_global_HRSWH.npz            # 全球海洋/陆地掩膜
│
├── cfg_global_ef.yaml               # EarthFormer 配置
│
├── data_parallel.py                 # 多 GPU BalancedDataParallel
│
├── EF_Global_Train.py               # 🌍 全球 SWH 训练脚本
├── EF_Global_Inference_EpochEnsemble.py  # 🌍 Epoch 集成推理脚本
│
└── README.md                        # 本说明文档
```

---

## 🧩 二、任务定义

### 🎯 预测目标
- **变量**：有效波高（Significant Wave Height, SWH）
- **单位**：米（m）
- **空间范围**：全球海洋（70°S–70°N）
- **空间分辨率**：0.5° × 0.5°（281 × 720）
- **时间分辨率**：逐小时 ERA5

### ⏱ 输入–输出设定
- **输入**：过去 **240 小时** 的全球 10 m 风场  
  - `u10`：zonal wind  
  - `v10`：meridional wind  
- **输出**：未来 **1 小时** 的全球 SWH 场

---

## 📊 三、数据来源与预处理

### 1️⃣ 数据来源
- **ERA5 再分析数据**
  - 风场：`u10`, `v10`
  - 波高：`swh`
- 数据格式：NetCDF (`.nc`)

示例路径：
```text
E:/Era5-Global-0.5/2000.nc
```

### 2️⃣ 空间处理
- 纬度范围：70°N → 70°S
- 经度范围：0° → 360°
- 网格大小：281 × 720

### 3️⃣ 周期边界处理（物理一致性）
- 经度方向采用 **周期扩展（Periodic Padding）**
- 左右各扩展 20 个格点，避免经度不连续导致的物理伪影

---

## 🧠 四、数据集构建（DynamicDataset）

- 使用过去 **240 h** 的风场作为输入
- 预测对应时间步的 **SWH**
- 数据组织形式：

```text
Input:  (T, H, W, C) = (240, 281, 720+40, 2)
Label:  (1, 281, 720, 1)
```

---

## 🧮 五、模型结构（EarthFormer + CNN）

整体采用 **CNN + Cuboid Transformer + CNN** 的混合架构：

```text
Wind Field (u10, v10)
        ↓
3D CNN Downsampling
        ↓
EarthFormer (Cuboid Transformer)
        ↓
2D CNN Upsampling
        ↓
Global SWH Prediction
```

### 🔹 1. 下采样模块（3D CNN）
- 时空联合卷积
- 降低计算成本
- 提取局地风浪动力学特征

### 🔹 2. EarthFormer（Cuboid Transformer）
- 轴向注意力（Axial Attention）
- 全局向量（Global Tokens）
- 长程时空依赖建模
- 有效刻画 **swell 传播与全球相关性**

### 🔹 3. 上采样模块（2D CNN）
- 恢复原始空间分辨率
- 输出单通道 SWH

---

## ⚙️ 六、训练策略（EF_Global_Train.py）

### 1️⃣ 损失函数（纬度加权 MSE）
引入 **纬度余弦权重**，缓解高纬区域网格密集导致的误差放大：

\[
\mathcal{L} = \cos(\varphi) \cdot (H_{pred} - H_{true})^2
\]

### 2️⃣ 海陆掩膜
- 使用 `mask_global_HRSWH.npz`
- 陆地格点在训练与评估中被 mask

### 3️⃣ 训练设置
- 优化器：AdamW
- 学习率：1e-3
- Batch size：12
- 多 GPU：BalancedDataParallel
- 训练年份：2000–2017
- 验证年份：2022

---

## 📈 七、验证与可视化

- TensorBoard 记录训练过程
- 空间 RMSE 热力图
- 逐 Epoch 模型保存

启动 TensorBoard：
```bash
tensorboard --logdir logs/
```

---

## 🔍 八、推理与 Epoch Ensemble

### Epoch Ensemble 思想
- 加载多个不同 Epoch 的模型
- 对同一输入进行预测
- 对输出取平均
- 提升预测稳定性与泛化能力

### 推理脚本
```bash
python EF_Global_Inference_EpochEnsemble.py
```

### 输出结果
- RMSE
- Relative RMSE
- Correlation coefficient
- Bias

结果保存于：
```text
EartherFormerResults/epochensemble/
```

---

## 💻 九、运行方式

### 1️⃣ 训练
```bash
python EF_Global_Train.py
```

### 2️⃣ 推理
```bash
python EF_Global_Inference_EpochEnsemble.py
```

---

## 🧱 十、运行环境

- OS：Ubuntu 20.04+
- GPU：RTX 3090 / 4090
- RAM：≥128 GB

依赖：
```bash
pip install torch torchvision xarray numpy matplotlib tqdm omegaconf scikit-learn
```

---

## 📚 十一、参考文献

1. Wang, Xinxin, and Haoyu Jiang. "Physics-guided deep learning for skillful wind-wave modeling." Science Advances 10.49 (2024): eadr3559.

2. Gao, Zhihan, et al. "Earthformer: Exploring space-time transformers for earth system forecasting." Advances in Neural Information Processing Systems 35 (2022): 25390-25403.

---

## 🔗 十二、代码引用

完整代码仓库地址：

👉 https://github.com/YulKeal/AI-Wave-Height-Model/tree/main/global_ocean

---

## 📝 说明

本项目用于科研与方法验证。使用或修改代码时，请注明数据来源与上述参考文献。
