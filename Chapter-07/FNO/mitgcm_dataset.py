import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
import torch.nn.functional as F


class MITgcmFNO(Dataset):
    """
    FNO专用的MITgcm数据集类
    用于加载低分辨率-高分辨率数据对进行超分辨率训练
    """

    def __init__(self, folder):
        """
        初始化数据集

        Args:
            folder: 包含.npz数据文件的文件夹路径
        """
        # 获取所有npz文件并按文件名排序
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError("No .npz files in dataset folder")

        print(f"Loaded {len(self.files)} samples from {folder}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.files)

    def __getitem__(self, idx):
        """
        获取单个数据样本

        Args:
            idx: 样本索引

        Returns:
            lr_up: 上采样后的低分辨率数据 (1, nz, nx)
            hr: 原始高分辨率数据 (1, nz, nx)
            lr: 原始低分辨率数据 (1, nz, nx//scale)
        """
        # 加载npz文件
        d = np.load(self.files[idx])
        hr = d["hr"]  # 高分辨率数据
        lr = d["lr"]  # 低分辨率数据

        # 转换为PyTorch Tensor并增加通道维度
        hr = torch.from_numpy(hr).float().unsqueeze(0)  # (1, nz, nx)
        lr = torch.from_numpy(lr).float().unsqueeze(0)  # (1, nz, nx//scale)

        # FNO要求输入输出尺寸相同，因此对LR进行上采样
        # 使用双三次插值将LR上采样到HR的尺寸
        lr_up = F.interpolate(
            lr.unsqueeze(0),  # 增加batch维度: (1, 1, nz, nx//scale)
            size=hr.shape[-2:],  # 目标尺寸: (nz, nx)
            mode="bicubic",  # 双三次插值
            align_corners=False
        )
        lr_up = lr_up.squeeze(0)  # 移除batch维度: (1, nz, nx)

        # 返回: 上采样LR, 真实HR, 原始LR
        return lr_up, hr, lr