import os
os.environ['CUDA_VISIBLE_DEVICES'] ="1" #选择GPU
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import stats
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from math import sqrt
import torch.nn.functional as F



class MLP_new(nn.Module):
    def __init__(self, input_channels=20, output_channels=1, hidden_dim=128):
        super(MLP_new, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_channels)
        )

    def forward(self, x):
        # x 的形状为 (batch_size, channels, length) = (64, 20, 32)
        batch_size, channels, length = x.size()
        # 调整形状以适应全连接层： (batch_size * length, channels)
        x = x.permute(0, 2, 1).contiguous().view(-1, channels)
        x = self.mlp(x)
        # 将形状调整回原来的形式： (batch_size, length, output_channels)
        x = x.view(batch_size, length, -1)
        # 重新排列维度为 (batch_size, output_channels, length)
        x = x.permute(0, 2, 1)
        return x  # 输出形状为 (64, 1, 32)

def edge_conv2d(im):
        # 用nn.Conv2d定义卷积操作
        conv_op = nn.Conv2d(3, 3, kernel_size=20, padding=1, bias=False).cuda()
        # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
        # 将sobel算子转换为适配卷积操作的卷积核
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        # 卷积核的第一个参数对应输出通道数量，这里我设置为3
        sobel_kernel = np.repeat(sobel_kernel, 20, axis=0)
        # 卷积核的第二个参数对应输入通道数量，这里我设置为3
        sobel_kernel = np.repeat(sobel_kernel, 20, axis=1)
        conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
        edge_detect = conv_op(im)
        return edge_detect

class UNetDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(UNetDownsample, self).__init__()
        # 卷积层，步长为2进行下采样
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class MultiDownsampleUNet(nn.Module):
    def __init__(self):
        super(MultiDownsampleUNet, self).__init__()
        # 三次下采样，通道数可自行定义（可以不变或递增）
        self.down1 = UNetDownsample(in_channels=10, out_channels=10)
        self.down2 = UNetDownsample(in_channels=10, out_channels=10)
        self.down3 = UNetDownsample(in_channels=10, out_channels=10)
        self.down4 = UNetDownsample(in_channels=10, out_channels=10)
        self.down5 = UNetDownsample(in_channels=10, out_channels=10)
        self.down6 = UNetDownsample(in_channels=10, out_channels=10)
        self.down7 = UNetDownsample(in_channels=10, out_channels=10)

    def forward(self, x):
        x = self.down1(x)  # 第一次下采样
        x = self.down2(x)  # 第二次下采样
        x = self.down3(x)  # 第三次下采样
        x = self.down4(x)  # 第四次下采样
        x = self.down5(x)  # 第五次下采样
        x = self.down6(x)  # 第六次下采样
        x = self.down7(x)  # 第七次下采样
        return x


class UNetUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(UNetUpsample, self).__init__()
        # 转置卷积用于上采样，步幅为2
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.relu(x)
        return x

# 定义上采样过程
class MultiUpsampleUNet(nn.Module):
    def __init__(self):
        super(MultiUpsampleUNet, self).__init__()
        self.up1 = UNetUpsample(in_channels=10, out_channels=10)
        self.up2 = UNetUpsample(in_channels=10, out_channels=10)
        self.up3 = UNetUpsample(in_channels=10, out_channels=10)  
        self.up4 = UNetUpsample(in_channels=10, out_channels=10)
        self.up5 = UNetUpsample(in_channels=10, out_channels=10)
        self.up6 = UNetUpsample(in_channels=10, out_channels=10)
        self.up7 = UNetUpsample(in_channels=10, out_channels=10)

    def forward(self, x):
        x = self.up1(x)  # 第一次上采样
        x = self.up2(x)  # 第二次上采样
        x = self.up3(x)  # 第三次上采样
        x = self.up4(x)  # 第四次上采样
        x = self.up5(x)  # 第五次上采样
        x = self.up6(x)  # 第六次上采样
        x = self.up7(x)  # 第七次上采样
        return x


class llama_model(nn.Module):
    """
    基于 LLaMA 的海温预测模型。

    整体思路：
    1）空间压缩：首先将输入海温序列 z，形状 [B, T=10, C=1, H=128, W=128]，
        通过多层卷积下采样（MultiDownsampleUNet）在空间维上压缩为 [B, T, 1, 1]，
        得到每个样本在 10 天时间轴上的紧凑数值表示 x_vec ∈ R^{B×10}。

    2）数值 → 语言空间重编程：
        - 使用 ReprogrammingLayer，将 x_vec 视作 query（[B, 1, 10]），
          与 LLaMA 词向量映射后的子词表 source_embeddings ∈ R^{1000×4096} 做交叉注意力，
          得到对齐到 LLaMA 表示空间的数值特征 enc_out ∈ R^{B×1×4096}。
        - 构造简单的物理任务描述 prompt（海表温度预测），
          将 prompt 的 token embedding 与 enc_out 在序列维度上拼接，
          一同输入到轻量化的单层 LLaMA 中进行高维非线性建模。

    3）语言空间 → 物理场预测：
        - 从 LLaMA 输出中裁剪对应数值 token 的 hidden state，并将其从 4096 维线性压缩到 64 维，
          再通过 MLP_new 将 [B, 64, 1] 映射为 [B, 10, 1]，得到 10 个“时间 token”的隐表示。
        - 将该表示视为 [B, 10, 1, 1] 的极小空间特征图，通过 MultiUpsampleUNet 逐步上采样回
          [B, 10, 128, 128]，作为 LLaMA 引导的预测场 y_pred。
        - 与原始特征 x 做残差融合 output = x + y_pred，保留观测场的低频结构，
          再将 output 展平为 [B, 10, H*W]，通过第二个 MLP_new 沿时间维（10→1）进行压缩，
          最终恢复为 [B, 1, 128, 128] 的下一天海表温度预测场。

    该结构可以看作：
        CNN 下采样（物理场 → 时间序列） +
        LLaMA 重编程建模（数值 → 语言空间 → 数值） +
        UNet 上采样与残差恢复（时间序列 → 物理场），
    既利用了大模型的表征能力，又保持了时空场预测的结构约束。
    """
    def __init__(self):
        super(llama_model, self).__init__()

        # -------------------------------------------------------------
        # 1) 多层下采样 / 上采样 UNet（用于把128×128特征压缩成1×1，再恢复）
        # 输入： [B, 10, 128, 128]
        # 输出下采样： [B, 10, 1, 1]
        # 输出上采样： [B, 10, 128, 128]
        # -------------------------------------------------------------
        self.downsample = MultiDownsampleUNet()
        self.upsample = MultiUpsampleUNet()

        # -------------------------------------------------------------
        # 2) 加载轻量化的 LLaMA 模型（只保留1层）
        #    - 强制num_hidden_layers=1降低计算
        #    - 仅使用 LLaMA 作为“数值到语言空间的非线性映射器”
        # -------------------------------------------------------------
        llm_model = 'LLAMA'
        if llm_model == 'LLAMA':
            # 读取配置
            self.llama_config = LlamaConfig.from_pretrained('/home/dataDisk/lcc/code/llm/llama7bhf')
            self.llama_config.num_hidden_layers = 1                      # 只用1层 encoder
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

            # 加载 LLaMA 模型权重
            self.llm_model = LlamaModel.from_pretrained(
                    '/home/dataDisk/lcc/code/llm/llama7bhf',
                    local_files_only=True,
                    config=self.llama_config,
                )

            # 加载 tokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(
                    '/home/dataDisk/lcc/code/llm/llama7bhf/tokenizer.model',
                    trust_remote_code=True,
                    local_files_only=True
                )

        # -------------------------------------------------------------
        # 3) 读取 LLaMA 的词向量矩阵 (vocab × 4096)
        #    后续通过 mapping_layer 压缩到更小的 token embedding
        # -------------------------------------------------------------
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]   # e.g., 32000
        self.num_tokens = 1000                            # 压缩后的token数量

        # -------------------------------------------------------------
        # 4) 将词向量从 [vocab_size → 1000] 做线性压缩
        #    得到更小的 1000×4096 词表矩阵，便于交叉注意力使用
        # -------------------------------------------------------------
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # -------------------------------------------------------------
        # 5) 数值输入特征 x_vec (维度10) —— LLaMA 词向量 (4096维)
        #    之间的 Reprogramming Cross-Attention
        #
        # target_embedding：输入数值特征，形状 [B, 1, 10]
        # source_embedding：压缩后的词向量 [1000, 4096]
        # 作用：让数值特征“对齐”到 LLaMA 语言表示空间
        # -------------------------------------------------------------
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=10,      # 数值特征长度
            n_heads=2,       # 注意力头数量（轻量化）
            d_llm=4096,      # LLaMA embedding 维度
            attention_dropout=0.1
        )

        # -------------------------------------------------------------
        # 6) 数值预测部分的 MLP
        #    mlp_layer1：64 → 10，用于生成10个空间token
        #    mlp_layer2：10 → 1，对时间维压缩，生成最终输出
        # -------------------------------------------------------------
        self.mlp_layer1 = MLP_new(input_channels=64, output_channels=10, hidden_dim=128)
        self.mlp_layer2 = MLP_new(input_channels=10, output_channels=1, hidden_dim=128)


    def forward(self, z):
        """
        输入 z: [B, T=10, C=1, H=128, W=128]
        输出： [B, 1, 128, 128]
        """

        # ----------------- Step 1: 下采样压缩空间 ---------------------
        B, T, C, H, W = z.shape

        x = z.squeeze(2)              # 去掉C维 → [B, 10, 128, 128]
        x_down = self.downsample(x)   # 多层downsample → [B, 10, 1, 1]

        # 将空间压到1×1后，取10个时间点作为向量输入 LLaMA
        x_vec = x_down.view(B, T)     # [B, 10]

        # ----------------- Step 2: Reprogramming Cross-Attention -----
        enc_out = x_vec.unsqueeze(1)  # [B, 1, 10] 作为 query

        # 压缩词向量，得到 source embeddings
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # source_embeddings: [1000, 4096]

        # 数值 → 语言空间对齐
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        # 输出 enc_out: [B, 1, 4096]

        # ----------------- Step 3: 构建 Prompt + 拼接到 LLaMA 输入 -----
        prompt = [
            "Sea surface temperature prediction. Forecast the next day given the past 10 days observations."
            for _ in range(B)
        ]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).input_ids.to(device)

        prompt_length = prompt.shape[1]                                # prompt 的 token 数
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt)

        # 拼接 prompt + 数值 embedding
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

        # ----------------- Step 4: 过 LLaMA ---------------------------
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :64]            # 降维：4096 → 64
        dec_out = dec_out[:, prompt_length:, :] # 只取数值对应的1个token

        # dec_out: [B, 1, 64] → [B, 64, 1]
        dec_out = dec_out.permute(0, 2, 1)

        # ----------------- Step 5: 生成10个空间token ------------------
        dec_out = self.mlp_layer1(dec_out)      # [B, 64,1] → [B,10,1]

        # reshape 成上采样输入： [B,10,1,1]
        dec_out_map = dec_out.view(B, T, 1, 1)

        # ----------------- Step 6: 空间恢复 ----------------------------
        y_pred = self.upsample(dec_out_map)     # 恢复到 128×128: [B,10,128,128]

        # ----------------- Step 7: 与原始特征融合 ----------------------
        output = x + y_pred                      # 融合： [B,10,128,128]

        # ----------------- Step 8: MLP 压缩时间维 ----------------------
        output_flat = output.view(B, T, -1)      # 展平空间 → [B,10,H*W]

        output_flat = self.mlp_layer2(output_flat)  # [B,1,H*W]

        # reshape 回图像形式
        output = output_flat.view(B, 1, H, W)    # [B,1,128,128]

        return output


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

# ========= 工具函数 =========
def fit_minmax(x):
    """在训练集上拟合 min/max（忽略 NaN）"""
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    # 防止极端情况
    if np.isclose(x_max, x_min):
        x_max = x_min + 1e-8
    return float(x_min), float(x_max)

def transform_minmax(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def inverse_minmax(x_norm, x_min, x_max):
    return x_norm * (x_max - x_min) + x_min

# ========= 1) 加载原始数据 =========
data = np.load('data.npy')  # 原始: (365, 1, 1, 128, 128)
print("原始数据形状:", data.shape)

# ========= 2) 构造“过去10天预测第1天”的样本 =========
seq_len = 10
X_list, Y_list = [], []
for i in range(len(data) - seq_len):
    X_list.append(data[i:i+seq_len])        # (10, 1, 1, 128, 128)
    Y_list.append(data[i+seq_len])          # (1, 1, 128, 128)
X = np.array(X_list).squeeze(2)             # → (N, 10, 1, 128, 128)
Y = np.array(Y_list).squeeze(2)             # → (N, 1, 128, 128)
print("构造后的数据集形状:", X.shape, Y.shape)  # 与你现有脚本一致:contentReference[oaicite:2]{index=2}

# ========= 3) 先切分，再在训练集上拟合归一化参数 =========
N = len(X)
train_end = int(N * 0.7)
val_end   = int(N * 0.9)

X_train_raw, Y_train_raw = X[:train_end], Y[:train_end]
X_val_raw,   Y_val_raw   = X[train_end:val_end], Y[train_end:val_end]
X_test_raw,  Y_test_raw  = X[val_end:], Y[val_end:]

# —— 拟合仅使用训练集（可以把输入与标签拼一起求全局 min/max）——
#   注意：这里对“原始数据”计算 min/max（未把 NaN 改 0），避免 NaN=0 影响统计量
train_stack = np.concatenate([X_train_raw.reshape(train_end, -1),
                              Y_train_raw.reshape(train_end, -1)], axis=1)
train_min, train_max = fit_minmax(train_stack)
print(f"训练集拟合的 min/max: {train_min:.4f}, {train_max:.4f}")

# ========= 4) NaN 统一置 0，再用训练集 min/max 归一化 =========
def nan_to_zero(a):
    a = a.copy()
    a[np.isnan(a)] = 0.0
    return a

X_train = nan_to_zero(X_train_raw)
Y_train = nan_to_zero(Y_train_raw)
X_val   = nan_to_zero(X_val_raw)
Y_val   = nan_to_zero(Y_val_raw)
X_test  = nan_to_zero(X_test_raw)
Y_test  = nan_to_zero(Y_test_raw)

# 用训练集的 min/max 对三份数据做同尺度归一化
X_train = transform_minmax(X_train, train_min, train_max)
Y_train = transform_minmax(Y_train, train_min, train_max)
X_val   = transform_minmax(X_val,   train_min, train_max)
Y_val   = transform_minmax(Y_val,   train_min, train_max)
X_test  = transform_minmax(X_test,  train_min, train_max)
Y_test  = transform_minmax(Y_test,  train_min, train_max)

# ========= 5) 转 Tensor =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
X_val   = torch.tensor(X_val,   dtype=torch.float32, device=device)
Y_val   = torch.tensor(Y_val,   dtype=torch.float32, device=device)
X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
Y_test  = torch.tensor(Y_test,  dtype=torch.float32, device=device)

# ========= 6) 定义 llama相关 模型 =========
model = llama_model().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ========= 7) 训练 + 验证 =========
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out_train = model(X_train)                  # 输出序列
    pred_train = out_train
    loss = criterion(pred_train, Y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_val = model(X_val)
        pred_val = out_val
        val_loss = criterion(pred_val, Y_val)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

# ========= 8) 测试：用训练集 min/max 反归一化后计算 RMSE =========
model.eval()
with torch.no_grad():
    out_test = model(X_test)
    pred_test_norm = out_test.cpu().numpy()
    Y_test_norm    = Y_test.cpu().numpy()

# 反归一化（严格使用训练集的 min/max）
pred_test = inverse_minmax(pred_test_norm, train_min, train_max)
Y_test_gt = inverse_minmax(Y_test_norm,    train_min, train_max)

rmse = sqrt(mean_squared_error(Y_test_gt.flatten(), pred_test.flatten()))
print(f"✅ 测试集 RMSE（使用训练集 min/max 反归一化）: {rmse:.4f}")