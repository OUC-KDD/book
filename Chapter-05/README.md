# 🌊 浮标数据基于 SVR 的海浪高度预测项目

本项目基于真实浮标观测数据（站点 46221、46251），采用 **滑动时间窗口 + 支持向量回归（SVR / LinearSVR）** 对海浪有效波高（WVHT）进行短期预测。

预测任务：

> **使用过去 24 小时的 WVHT + 风场数据（u10、v10），预测下一小时的 WVHT。**

模型采用 **StandardScaler + LinearSVR**，适合处理超过 4 万条的大规模时间序列样本，可在数秒内完成训练。

---

# 📁 项目结构

```
project/
│
├── 46221.csv                     # 浮标站点 46221 原始数据
├── 46251.csv                     # 漂浮站点 46251 原始数据
│
├── svr.py                        # 主程序（数据预处理 + 滑动窗口 + SVR 训练与测试）
└── README.md                     # 项目说明文档（本文件）
```

---

# 🧩 功能概述

本项目主要包括：

1. **浮标数据预处理**
2. **特征构造（24h × 3变量 = 72维）**
3. **SVR 模型训练与测试**
4. **评价指标（RMSE/MAE/R²）**

---

# 📊 一、数据预处理

## 1. 数据字段说明
| 列名 | 含义 |
|------|------|
| DateTime | 时间戳 |
| WVHT | 有效波高（单位：米） |
| u10 | 10 米高度风向东西分量 |
| v10 | 10 米高度风向南北分量 |

## 2. 时间序列重采样与缺失修复
- 按 1 小时重采样
- 线性插值修复缺失值
- 前向/后向填充首尾数据

## 3. 滑动窗口构造
```
history_len = 24    # 输入过去24小时
horizon = 1         # 预测下一小时
X.shape = (43800, 72)
y.shape = (43800,)
```

---

# 🧠 二、SVR 模型构建

### 模型结构

```python
SVR_Pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', LinearSVR(
        C=1.0,
        epsilon=0.1,
        max_iter=5000
    ))
])
```

---

# ⚙️ 三、训练与测试流程

## 数据集划分
- **训练集：80%**
- **测试集：20%**

## 评价指标

### RMSE
\[
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}
\]

### MAE
\[
MAE = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|
\]

### R²
\[
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
\]

---

# 🧾 四、实验结果（46221 浮标）

```
=== Train ===
R^2 : 0.9525
MAE : 0.0559
MSE : 0.0059
RMSE: 0.0769

=== Test ===
R^2 : 0.9677
MAE : 0.0596
MSE : 0.0074
RMSE: 0.0861
```

**结果说明：**

- 测试集 R² ≈ **0.97**
- RMSE ≈ **8.6 cm**
- 预测精度优秀，能有效捕捉 WVHT 的时序变化趋势

---

# 💻 五、运行说明

### 安装依赖
```bash
pip install numpy pandas scikit-learn
```

### 运行主程序
```bash
python svr.py
```

---

# 🧱 六、环境需求
- Python ≥ 3.8
- pandas ≥ 1.3
- numpy ≥ 1.20
- scikit-learn ≥ 1.0

---

# 📚 七、项目亮点

- 高效的大规模浮标时间序列预测方法  
- 滑动窗口 + SVR 的经典时间序列建模流程  
- 适合作为海洋工程/机器学习课程项目示例  

---

# 📚 九、引用

1, Cortes, C., and Vapnik, V. 1995. Support-vector networks. Machine Learning, 20(3), 273–297. DOI: https://doi.org/10.1007/BF00994018

2, Smola, A. J., and Schölkopf, B. 2004. A tutorial on support vector regression. Statistics and Computing, 14, 199–222. DOI: https://doi.org/10.1023/B:STCO.0000035301.49549.88

---


