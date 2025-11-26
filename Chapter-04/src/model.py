import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ViT(nn.Module):
    """
    Vision Transformer Backbone for Global Attention [cite: 42-43]
    """
    def __init__(self, image_size, patch_size, dim, depth, heads, channels=3, dim_head=64, dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.transformer = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.reshape = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                                 p1=patch_height, p2=patch_width, h=image_height // patch_height)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = self.transformer(x)
        x = self.reshape(x)
        return x

def shift_operation(y, n=6):
    """
    Shift Convolution Logic [cite: 120-122]
    """
    B, C, H, W = y.shape
    num = C // n
    out = torch.zeros_like(y)
    out[:, num * 0:num * 1, 1:, :] = y[:, num * 0:num * 1, :-1, :]  # shift down
    out[:, num * 1:num * 2, :-1, :] = y[:, num * 1:num * 2, 1:, :]  # shift up
    out[:, num * 2:num * 3, :, :-1] = y[:, num * 2:num * 3, :, 1:]  # shift left
    out[:, num * 3:num * 4, :, 1:] = y[:, num * 3:num * 4, :, :-1]  # shift right
    out[:, num * 4:, :, :] = y[:, num * 4:, :, :]  # no shift
    return out

class CSC(nn.Module):
    """
    Parallel Convolution Module (Shift Conv)
    """
    def __init__(self):
        super(CSC, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 1, 1)
        self.conv2 = nn.Conv2d(12, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = shift_operation(x)
        x = self.conv2(x)
        return x

class SGU(nn.Module):
    """
    Gated Feed-Forward Network (GFFN) [cite: 160-165]
    """
    def __init__(self, patch_size):
        super(SGU, self).__init__()
        self.catConv = nn.Conv2d(6, 3, kernel_size=1)
        self.norm1 = nn.LayerNorm([3, patch_size, patch_size])
        self.conv = nn.Conv2d(3, 8, 1, 1)
        self.project_in = nn.Conv2d(3, 16, kernel_size=1)
        self.dwconv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16, bias=False)
        self.project_out = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x):
        catOut = self.catConv(x)
        x = catOut
        x = self.norm1(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2  # Gating mechanism
        catOut = self.conv(catOut)
        x = x + catOut
        x = self.project_out(x)
        return x

class CAMixer(nn.Module):
    def __init__(self, patch_size=9):
        super(CAMixer, self).__init__()
        self.vit = ViT(image_size=patch_size, patch_size=3, dim=27, depth=6, heads=4)
        self.csc = CSC()
        self.sgu = SGU(patch_size)
        self.linear1 = nn.Linear(patch_size * patch_size * 3, 20)
        self.linear2 = nn.Linear(20, 2) # Binary classification (Changed/Unchanged)

    def forward(self, img):
        vitOut = self.vit(img)
        cscOut = self.csc(img)
        catOut = torch.cat((vitOut, cscOut), 1) # Parallel fusion [cite: 158]
        x = self.sgu(catOut)
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out