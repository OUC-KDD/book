# 🌊 I2SB海表温度补全项目

本项目基于 I2SB（Image-to-Image Schrödinger Bridge） 思想，将其应用于 海表温度（SST）云遮挡重建。
在SST 数据集设置下，本代码将 完整 SST 场 $x_0$ 与 被云遮挡/噪声污染的 SST 场 $x_1$ 视为端点分布，通过 Schrödinger Bridge 扩散过程学习从 $x_1$ → $x_0$ 的最优输运重建路径。
本实验使用 真实云掩模 和 高斯噪声模拟不同 N/S 比 构建训练集，并在真实的 SST 数据上进行实验。
核心特点：

端点驱动的扩散过程：同时依赖目标场 $x_0$ 与损坏场 $x_1$；

对称 $\beta$ 调度：构造前向 / 反向标准差，显式实现 Schrödinger Bridge；

专为 SST 修复设计的损失与评估：RMSE / MAE / MSE / R² / SSIM / PSNR。

---

## 📁 项目结构

```
project/
│
├── New_data/FNO_based_2024_2/data/      # 预处理好的 SST 数据（与 DINCAE 相同）
│   ├── 0.1_Cloud_mask_8_train_South_Sea_miss.h5
│   ├── 0.1_Cloud_mask_8_train_South_Sea_true.h5
│   └── ...
├── data/                     # 存放原始 NetCDF 文件（按月份分文件夹）
│   ├── 2022_01/nc
│   │   ├── MUL_OPER_SST_L4A_FU_12H_20220101T12_dps_050_10_sst.nc
│   │   ├── ...
│   └── ...
├── Visual_Tool.py                   # 数据提取与预处理
│
├── run.py                               # I2SB-SST 主训练脚本（DSBRunner）
├── diffusion.py                         # Schrödinger Bridge 扩散过程 (Diffusion / AsyncDiffusion)
├── model.py                             # Image64Net + EMA (I2SB 噪声预测网络)
├── utils.py                             # beta 调度、时间步采样等工具函数
├── deal_sst_util.py                     # SST 数据读取工具（read_cache 等）
├── mask_obtain.py                       # 云掩模读取与生成
├── reconstru_visual.py                  # 重建结果可视化 (visua_and_save)
└── README.md                            # 项目说明文档


```

---

## 🧩 功能概述

整个项目可以分为两大部分：

### 1、数据预处理（Visual_Tool.py + deal_sst_util.py）

从原始 NSOAS L4 NetCDF 文件中提取南海区域；

根据 WHU 云掩模与设定的 N/S 比构造“缺失 + 噪声”的 SST 样本；

按 DINCAE 同样的规范输出训练 / 验证用的 .h5 文件。

### 2、I2SB-SST 训练与重建（run.py + diffusion.py + model.py）

使用 Schrödinger Bridge 扩散过程，将损坏图 $x_1$ 映射回完整图 $x_0$；

在云遮挡区域内实现 SST 重建，并输出 RMSE / MAE / R² / SSIM / PSNR 等指标。

---

## 📊 一、数据处理（`Visual_Tool.py`）

### 1. 数据来源与区域设置

原始数据位于：

```data/YYYY_MM/nc/*.nc```


数据来源：

数据源：国家卫星海洋应用服务（NSOAS）SST 四级产品（Mulit-mission L4）；

时间范围：2022年1月 – 2023年4月；

空间范围：南海局地海域

纬度：23°N – 26°N

经度：110°E – 113°E

空间分辨率：约 5 km × 5 km；

这些原始 L4 数据本身无缺失，作为 $x_0$ 的“完整真值”。

### 2. 数据裁剪与构造缺失（Visual_Tool.py）

```Visual_Tool.py``` 用于：

从原始 NetCDF (.nc) 中读取变量：

lon / lat：经纬度

sst：海表温度场

提取南海目标区域（23–26°N，110–113°E），并将每日 / 12h SST 裁剪并栅格化。

根据设定的云掩模与 N/S 比参数，生成带缺失/噪声的 SST 图像：

```
mask_type = "Cloud_mask"：从 WHU Cloud Dataset 中读取真实云掩模；

corrup_rate ∈ {8, 25, 46, 68}：控制云覆盖比例；

N_S_ratio ∈ {0.1, 0.2, 0.3,...}：生成标准正态噪声模拟不同噪声水平（与 DINEOF 一致）。
```

输出至：

```
New_data/FNO_based_2024_2/data/
├── {N_S_ratio}_{mask_type}_{corrup_rate}_train_South_Sea_miss.h5
├── {N_S_ratio}_{mask_type}_{corrup_rate}_train_South_Sea_true.h5
├── {N_S_ratio}_{mask_type}_{corrup_rate}_valid_South_Sea_miss.h5
├── {N_S_ratio}_{mask_type}_{corrup_rate}_valid_South_Sea_true.h5
└── ...
```

其中：

*_miss.h5：带云掩模 + 噪声的 SST；

*_true.h5：对应的完整 SST 真值。

运行示意（根据你自己的接口调整）：

```
python Visual_Tool.py
```

生成完 .h5 后，即可用于 I2SB /  其他模型的对比实验。


## 🧠 二、Schrödinger Bridge 扩散过程（diffusion.py）

Diffusion 类实现了 I2SB 中的 Schrödinger Bridge：

### 1. β 调度与标准差构造

在 run.py 中通过：

```
betas = make_beta_schedule(n_timestep=opt.interval,
                           linear_end=opt.beta_max / opt.interval)
betas = np.concatenate([betas[:opt.interval//2],
                        np.flip(betas[:opt.interval//2])])
diffusion = Diffusion(betas, opt.device)

```

构造对称的 β 序列，随后在 Diffusion.__init__ 中计算：

```std_fwd```: 前向累积标准差

```std_bwd```: 反向累积标准差

```mu_x0, mu_x1, std_sb```: 由高斯乘积得到的桥接系数

对应理论上：

```
𝑞
(
𝑥
𝑡
∣
𝑥
0
,
𝑥
1
)
=
𝑁
(
𝑥
𝑡
∣
𝜇
0
𝑥
0
+
𝜇
1
𝑥
1
,
𝜎
SB
2
)
q(x
t
	​

∣x
0
	​

,x
1
	​

)=N(x
t
	​

∣μ
0
	​

x
0
	​

+μ
1
	​

x
1
	​

,σ
SB
2
	​

)
```
### 2. 前向采样：q_sample(step, x0, x1)

训练时，对给定 $x_0$（完整 SST）和 $x_1$（损坏 SST）采样中间状态：

```
xt = diffusion.q_sample(step, x0, x1)
```

### 3. 后验采样与 DDPM 过程

验证 / 推理阶段，通过：

```
xs, pred_x0s = diffusion.ddpm_sampling(
    steps=steps,
    pred_x0_fn=pred_x0_fn,
    x1=x1,
    mask=mask,
    ot_ode=opt.ot_ode,
    log_steps=log_steps,
    verbose=True
)
```

实现：

从损坏图 x1 沿 Schrödinger Bridge 逐步去噪，同时保持非云区强约束；

最终 ```pred_x0s[:, -1]``` 作为 I2SB-SST 的重建结果。

## 🧪 三、I2SB-SST 模型与训练（run.py + model.py）

### 1. 数据加载与划分

在 ```run.py``` 中，首先读取 .h5：

```
x_train, y_train, x_valid, y_valid = deal_sst_util.read_cache(
    f'../New_data/FNO_based_2024_2/data/{opt.N_S_ratio}_{opt.mask_type}_{opt.corrup_rate}_train_{opt.save_file}_miss.h5'
)

train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
val_data   = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
```

其中：

```x_*```：带缺失 / 噪声的 SST 序列；

```y_*```：对应完整 SST 序列。

在训练中，具体使用：

```
x1 = batch_x[:, 6:7]   # 损坏 SST 图像
x0 = batch_y[:, 6:7]   # 真实 SST 图像
```

### 2. 模型初始化

```
noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
net = Image64Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
ema = ExponentialMovingAverage(net.parameters(), decay=opt.ema)
```

```Image64Net```：I2SB 噪声预测网络，输入 ```(x_t, step, cond=x1)```；

```EMA```：保存模型的滑动平均版本用于采样。

### 3. 训练目标（噪声回归）

每个 batch 中：

采样时间步：

```
step = torch.randint(0, opt.interval, (x0.shape[0],), device=opt.device)
```

搜取桥上状态：

```
xt = diffusion.q_sample(step, x0, x1)
```

构造标签（I2SB 噪声）：

```
std_fwd = diffusion.get_std_fwd(step, xdim=x0.shape[1:])
label   = (xt - x0) / std_fwd
```

网络预测：

```
pred    = net(xt, step, cond=x1)
loss    = MSE(pred, label)
```

反向传播：

```
loss.backward()
optimizer.step()
ema.update()
```

在训练过程中会定期调用 ```reconstru_visual.visua_and_save``` 保存：

损坏样本：miss6

重建结果：recons

对应真值：ground_recons / valid_ground_recons
##  💻 四、验证与指标评估

验证阶段使用 EMA 网络进行 DDPM 式采样：

```
xs, pred_x0s = runner.run_ddpm_sampling(
    opt,
    x1,
    cond=x1,
    clip_denoise=opt.clip_denoise,
    log_count=1,
    nfe=20
)
reconstructed_image = pred_x0s[:, -1]
```

对 云掩模内部区域 计算多种指标：

- MSE / RMSE

- MAE

- R²

- SSIM (pytorch_ssim)

- PSNR

并记录：

- batch 最优（最低 RMSE / MAE / MSE，最高 R² / SSIM / PSNR）；
- 各 epoch 平均指标的最优值；

将最终结果写入：```result_{N_S_ratio}_{corrup_rate}```文件中方便对比不同云覆盖率 / N/S 比的性能。

## 🔧 五、运行说明
### 第一步：从原始 NetCDF 生成 h5 数据

确认：

原始 NetCDF 在 ```data/YYYY_MM/nc/*.nc``` 中；

在 ```Visual_Tool.py``` 中设置好：

时间范围（2022.01–2023.04）

区域范围（23–26°N，110–113°E）

```N_S_ratio、mask_type="Cloud_mask"、corrup_rate``` 等参数。

执行：

```
python Visual_Tool.py
```

生成对应的：```New_data/FNO_based_2024_2/data/{N_S_ratio}_{mask_type}_{corrup_rate}_train_South_Sea_*.h5```

### 第二步：运行 I2SB-SST 训练与验证

```run.py``` 已内置多组实验循环：

```
for N_S_ratio in {0.2, 0.3}:
    for corrup_rate in {68,46,25,8}:
        ...
        runner = DSBRunner(opt, log)
        runner.train(opt, train_data, val_data)
```

直接执行：
```
python run.py
```

程序会：

读取对应 .h5 数据；

进行 I2SB 训练与验证；

自动输出重建可视化图像与定量评估结果。

## 🧱 六、环境依赖

建议环境：

Python ≥ 3.8

PyTorch ≥ 1.10

依赖建议：

```
pip install numpy matplotlib tqdm easydict ipdb
pip install h5py netCDF4
pip install scikit-image opencv-python
pip install pytorch-ssim
```

如需地理可视化，可再安装 ```Basemap``` / ```Cartopy``` 等库（按实际需求）。

## 📚 七、引用
1, Liu, G.-H., Vahdat, A., Huang, D.-A., Theodorou, E. A., Nie, W., & Anandkumar, A. (2023). I²SB: image-to-image Schrödinger bridge. In Proceedings of the 40th International Conference on Machine Learning (pp. 22042-22062). ACM. DOI: https://doi.org/10.5555/3618408.3619233.

2, I2SB. https://github.com/NVlabs/I2SB