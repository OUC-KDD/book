# 联邦学习 FedAvg（Python）方法

本项目提供一个基于 **联邦平均（FedAvg）** 的分布式机器学习方法实现，用于在**保护数据隐私的前提下进行模型训练**。学生可直接运行：

```
python main_fed.py
```

即可训练模型。

---

## 🧩 功能概述

本代码实现了联邦平均（Federated Averaging，FedAvg）算法，这是一种经典的联邦学习框架，通过以下方式实现分布式模型训练：

- 多个客户端在本地训练模型（不上传原始数据）
- 客户端将训练后的模型参数发送给服务器
- 服务器聚合所有客户端的模型参数（加权平均）
- 服务器将聚合后的模型参数分发给客户端，重复训练过程

核心公式（模型参数聚合）：

\[
  w_{t+1}=\sum_{k=1}^K\frac{n_k}{n}w_k^t
\]


---

## 📁 项目结构


```
.
├── main_fed.py                  # 主程序
├── data/
│   └── MNIST                    # 数据集
├── models/
│   └── Fed.py                   # 模型平均
│   ├── Net.py                   # 定义网络类型
│   └── update.py                # 本地更新    
├── utils/
│   └── options.py               # 参数设置
│   └── sampling.py              # 对Non-IID的数据分布进行采样 
└── README.md                    # 本文档
```

---

## ⚙️ 一. 环境依赖

建议使用 Python 3.6以上。

### Conda 快速安装：

```
conda create -n pinn python=3.10
conda activate fedavg

pip install torch numpy scipy pyDOE
```

若要使用 GPU，请安装支持 CUDA 的 PyTorch 版本。

---

## ▶️ 二. 运行教程

确保数据集已正确分区到 ./data/mnist/ 下。

### 直接运行：

```
python main_fed.py
```

程序将自动：

1. 下载并划分数据集  
2. 初始化客户端和服务器  
3. 执行联邦训练过程（多轮通信）
4. 自动保存全局模型参数 


---

## 🧠 三. 代码说明（核心部分）

### 1. 神经网络结构

- 基础模型：简单卷积神经网络（CNN）
- 输入：28×28 灰度图像（MNIST）
- 输出：10 个类别（数字 0-9）

### 2. 联邦训练流程
- 基础模型：简单卷积神经网络（CNN）
- 本地训练：客户端下载全局模型，用本地数据训练
- 参数上传：客户端将训练后的模型参数发送给服务器
- 参数聚合：服务器按 FedAvg 算法聚合客户端参数
- 模型更新：服务器更新全局模型，进入下一轮训练

### 3. 性能评估
- 每轮训练后在测试集上评估全局模型准确率
- 记录训练过程中的损失值和准确率变化

## 📚 四、引用
[McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.](https://www.semanticscholar.org/paper/Communication-Efficient-Learning-of-Deep-Networks-McMahan-Moore/d1dbf643447405984eeef098b1b320dee0b3b8a7)

[Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo.](http://doi.org/10.5281/zenodo.4321561)