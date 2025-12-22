# 🌊 DINCAE海表温度补全项目

本项目基于 DINCAE（Data-Interpolating Convolutional Auto-Encoder）思路，通过卷积自编码器对海表温度（SST）进行 缺失区域重建。
本实验使用 真实云掩模 和 高斯噪声模拟不同 N/S 比 构建训练集，并在真实的 SST 数据上进行实验。

---

## 📁 项目结构

```
project/
├── New_data/
│ ├── L4_12H/
│ │ ├── 2022_01/nc/.nc
│ │ ├── 2022_02/nc/.nc
│ │ └── ...
│ └── ...
│
├── data/ # 中间缓存数据（由脚本生成/读取）
│ ├── South_Sea_Train_real.h5 # 训练期真实 SST（无缺失）
│ ├── South_Sea_Test_real.h5 # 测试期真实 SST（无缺失）
│ ├── South_Sea_Train_time.h5 # 时间索引（与周序列对齐）
│ └── South_Sea_Test_time.h5
│
├── dincae_train.py # 主运行脚本
├── deal_sst_util.py # NC/H5 读写、变量提取、NaN/0 处理等工具函数
├── mask_obtain.py # 云掩模读取与生成（Cloud_mask + corrup_rate）
└── README.md
```

---

 🧩 功能概述

本项目包含三个核心部分：

1) 数据准备（原始 NC 下载并放置）

- 原始 SST L4 NetCDF 文件需从 NSOAS 平台下载，并存入 “New_data/L4_12H/”。

2) 数据组织与样本构造（dincae_train.py）

- 读取训练/测试期真实 SST（h5 缓存）。
- 基于时间窗构造周序列（7 天为一个样本），并选取第 7 天（索引 6）的 SST 作为重建目标。
- 生成 "missing / mask / lon / lat / time" 等 DINCAE 所需输入。

3) DINCAE 重建与评估（DINCAE 包内部）

- 在 "DINCAE.reconstruct_gridded_nc(...)" 内完成模型训练、验证、推理与保存结果。

---

## 📦 一、原始数据下载

1. 数据来源

- 数据平台：NSOAS 专题数据下载
- 下载地址：https://osdds.nsoas.org.cn/SpecialSubject
  你需要自行从该网站下载 Multi-mission L4 SST（12H）NetCDF数据文件。

2. 数据存放位置（重要）
   请将下载的 .nc文件按月份/目录组织后放到：

```
New_data/L4_12H/
└── 2022_01/
└── nc/
├── *.nc
└── ...
└── 2022_02/
└── nc/
├── *.nc
└── ...
```

## 🗂️ 二、缓存数据（data/ 目录）说明

代码直接读取以下文件：

- ./data/South_Sea_Train_real.h5
- ./data/South_Sea_Test_real.h5
- ./data/South_Sea_Train_time.h5
- ./data/South_Sea_Test_time.h5

## ☁️ 三、缺失与噪声设置（Cloud_mask + N/S Ratio）

1. 云掩模（mask_obtain.py）
   脚本通过：

```
missing = 1 - mask_obtain("mask", mask_type, corrup_rate, mask_num=0)
```

生成缺失区域，其中：

```
mask_type = "Cloud_mask"
corrup_rate ∈ {8, 25, 46, 68} 表示云覆盖比例档位
``
并且将 corrup_rate 映射到一个字符串编号：
```

if corrup_rate == 8:  mask_path = "531"
if corrup_rate == 25: mask_path = "526"
if corrup_rate == 46: mask_path = "732"
if corrup_rate == 68: mask_path = "455"

```
使得 mask_obtain.py 内部应能根据 (mask_type, corrup_rate, mask_num) 找到对应掩模文件/掩模样本。

2. 噪声强度（N/S 比）
通过循环指定：
```

for N_S_ratio in {0.3}:
    for corrup_rate in {8}:
        run(...)

```
该参数会被传入：
```

DINCAE.reconstruct_gridded_nc(..., mask_path, N_S_ratio, corrup_rate)

```
最终由 DINCAE 内部决定如何注入噪声或如何在损失中体现 N/S 约束。

## 🧠 四、输入样本组织逻辑（你脚本的关键点）
程序将日尺度 SST 组成 7 天一个序列：
```

sw_width = 7
sequence_y = train_real_data[sw_width*i : sw_width*i + sw_width]

```

然后只取每个序列的第 7 天（索引 6）作为训练/测试目标场：
```

train_data_all = train_seq_all_y[:, 6]
test_data_all  = test_seq_all_y[:, 6]
train_time_all = train_all_time[:, 6]
test_time_all  = test_all_time[:, 6]

```

## 🚀 五、运行方式
1, 运行 DINCAE（训练 + 测试）
在项目根目录执行：

```

python dincae_train.py

```

脚本会自动输出：PyTorch / CUDA 可用性、数据形状与缺失率统计、训练/测试组织后的样本维度

并进入 DINCAE.reconstruct_gridded_nc(...) 执行主流程

2, 输出目录

脚本指定输出目录：

```

outdir = "output_file_0801"

```


## 🧱 六、环境依赖

建议环境：

Python ≥ 3.8

PyTorch ≥ 1.10

numpy

netCDF4

h5py（若 deal_sst_util 读写 h5 用到）

matplotlib（可选）

安装示例：

pip install numpy torch netCDF4 h5py matplotlib



📚 七、引用
A. Barth, A. Alvera-Azcrate, M. Licer, and J. M. Beckers, “Dincae1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations,” Geoscientific Model
Development, vol. 13, no. 3, pp. 1609–1622, 2020.
S. Ji, P. Dai, M. Lu, and Y. Zhang, “Simultaneous cloud detection and removal from bitemporal remote sensing images using cascade convolutional neural networks,” IEEE Transactions on Geoscience and
Remote Sensing, vol. 59, no. 1, pp. 732–748, 2021.
```
