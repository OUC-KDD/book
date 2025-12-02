

# 傅里叶神经算子（FNO）求解 Darcy 方程（Python）

本项目提供一个基于 **傅里叶神经算子（Fourier Neural Operator, FNO）** 的深度学习方法，用于求解 **Darcy 方程**。可直接运行：
```
python fourier_2d.py
```

即可训练模型。

---

## 📌 1. 功能简介

本代码使用傅里叶神经算子方法，学习从参数场（如渗透率场）到解场（如压力场）的映射关系，用于求解以下 Darcy 方程：

$$ 
\nabla\cdot(a(x)\nabla u(x))=f(x)\\
u(x)=0 
$$


其中：

- 输入：扩散系数场 $a(x)$
- 输出：流体压力场 $u(x)$
- 使用傅里叶变换在频域中进行卷积操作
- 优化器：Adam

---

## 📂 2. 项目结构

```
.
├── fourier_2d.py                         # 主程序
├── utilities.py  
├── data/
│   ├── piececonst_r421_N1024_smooth1.mat    # 训练数据
│   ├── piececonst_r421_N1024_smooth2.mat    # 测试数据
├── Darcy_Equation/
│   └── models/                     # 模型存储目录
│       ├── darcy_fno.pth       
└── README.md                       # 本文档
```

---

## ⚙️ 3. 环境依赖

建议使用 Python 3.9–3.10。


若要使用 GPU，请安装支持 CUDA 的 PyTorch 版本。

---

## ▶️ 4. 运行教程

确保数据已放在 `./data/` 下。

### 直接运行：

```
python main.py
```

程序将自动：

1. 加载 Darcy 方程数据集（参数场与解场） 
2. 构建 FNO 网络结构
3. 使用 Adam 优化器训练模型 
4. 自动保存模型参数  

---

## 🧠 5. 代码说明（核心部分）

### ✔️ FNO结构

- 输入参数场 $a(x)$
- 编码器：将输入映射到高维表示
- 傅里叶层：在频域进行线性变换 + 激活
- 解码器：将特征映射回物理空间
- 输出解 $u(x)$

### ✔️ 损失函数

使用归一化均方误差（NMSE）作为损失函数



### ✔️ 使用 Adam 训练

采用 PyTorch 内置的 Adam 优化器。

---

## 💾 6. 模型保存


完成训练后模型保存为：
  ```
  Darcy_Equation/models/darcy_fno.pth
  ```

---

## 📈 7. 引用

[Zongyi Li, et al. Fourier Neural Operator for Parametric Partial Differential Equations. International Conference on Learning Representations. 2021.](https://openreview.net/forum?id=c8P9NQVtmnO)

[Zongyi Li. FNO PyTorch Implementation.](https://github.com/neuraloperator/neuraloperator)

### 数据来源
[Zongyi Li, et al. Dataset.](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
将下载好的数据集解压并放置于data文件夹中即可