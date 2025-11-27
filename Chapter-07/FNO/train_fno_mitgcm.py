import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from mitgcm_dataset import MITgcmFNO
from neuralop.models.fno import FNO
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device =", device)

# ---------------------------
# Dataset loading
# ---------------------------
dataset = MITgcmFNO("./data/dataset/")
n_total = len(dataset)

n_train = int(n_total * 0.8)
n_val   = int(n_total * 0.1)
n_test  = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=2, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=False)  # 测试用 batch=1 最合适

# ---------------------------
# Model
# ---------------------------
model = FNO(
    n_modes=(12, 12),
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    n_layers=4,
    positional_embedding="grid"
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# Visualization helper
# ---------------------------
def save_vis(lr, hr, pred, fname="vis.png"):
    lr = lr[0,0].cpu().numpy()
    hr = hr[0,0].cpu().numpy()
    pred = pred[0,0].cpu().numpy()

    plt.figure(figsize=(12,4))
    titles = ["LR", "HR True", "FNO Prediction"]
    datas  = [lr, hr, pred]

    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(datas[i], aspect='auto', cmap="RdBu_r")
        plt.title(titles[i])
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ---------------------------
# Training loop
# ---------------------------
best_val = 9999.0
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, 201):
    # -------- train --------
    model.train()
    train_loss = 0

    for lr, hr, _ in train_dl:
        lr = lr.to(device)
        hr = hr.to(device)

        pred = model(lr)

        loss = criterion(pred, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    # -------- val --------
    model.eval()
    val_loss = 0
    example_batch = None

    with torch.no_grad():
        for lr, hr, l_res in val_dl:
            lr = lr.to(device)
            hr = hr.to(device)
            pred = model(lr)
            loss = criterion(pred, hr)
            val_loss += loss.item()

            if example_batch is None:
                example_batch = (l_res, hr, pred)

    val_loss /= len(val_dl)

    print(f"[Epoch {epoch:03d}] Train={train_loss:.6f}  Val={val_loss:.6f}")

    # -------- save best model + vis --------
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "checkpoints/fno_best.pth")
        print(f"  ✓ Saved best model  (val={best_val:.6f})")

        lr, hr, pred = example_batch
        save_vis(lr, hr, pred, fname=f"checkpoints/epoch{epoch:03d}_vis.png")

# -------- Final save --------
torch.save(model.state_dict(), "checkpoints/fno_final.pth")
print("训练完成。")

# ===========================================================
#                    ★ TEST EVALUATION ★
# ===========================================================
print("\n============= Running TEST evaluation =============")

model.eval()
test_loss = 0

with torch.no_grad():
    for i, (lr, hr, l_res) in enumerate(test_dl):
        lr = lr.to(device)
        hr = hr.to(device)
        pred = model(lr)

        loss = criterion(pred, hr)
        test_loss += loss.item()

        save_vis(l_res, hr, pred, fname=f"checkpoints/test_{i}_vis.png")

test_loss /= len(test_dl)

print(f"★ Test Loss = {test_loss:.6f}")

# 保存到文本文件
with open("checkpoints/test_metrics.txt", "w") as f:
    f.write(f"Test MSE = {test_loss:.6f}\n")

print("测试评估完成，并已保存可视化与误差文件。")