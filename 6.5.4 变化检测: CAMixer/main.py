import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from src.model import CAMixer
from src.dataset import DataManager, SARDataset
from src.utils import evaluate, postprocess

def train_model(model, train_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4) # lr from [cite: 363] approx
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix({'Loss': total_loss/len(train_loader), 'Acc': correct/total})

def inference(model, data_manager, device):
    model.eval()
    h, w = data_manager.h, data_manager.w
    pre_lab = data_manager.preclassify_lab
    
    # Generate test patches for ALL pixels
    x_test = data_manager.create_testing_patches()
    
    final_map = np.zeros((h, w))
    
    # Logic: Only use Deep Learning for uncertain pixels (label 1.5)
    # OR pixels where we want to refine.
    # The paper implies using it to refine results.
    
    print("Starting Inference...")
    with torch.no_grad():
        for i in tqdm(range(h)):
            for j in range(w):
                # If pre-classification is certain, use it (optional optimization)
                # But typically we might want to run DL on everything or just uncertain ones.
                # Here follows the notebook logic: Only predict uncertain pixels
                if pre_lab[i, j] != 1.5:
                    final_map[i, j] = pre_lab[i, j] - 1 # 1->0, 2->1
                else:
                    # Extract patch for (i,j)
                    idx = i * w + j
                    patch = x_test[idx]
                    patch_tensor = torch.FloatTensor(patch).unsqueeze(0).to(device)
                    
                    output = model(patch_tensor)
                    pred = output.argmax(dim=1).cpu().item()
                    final_map[i, j] = pred

    return final_map

def main():
    parser = argparse.ArgumentParser(description="CAMixer for SAR Change Detection")
    parser.add_argument('--im1', type=str, required=True, help='Path to Image T1')
    parser.add_argument('--im2', type=str, required=True, help='Path to Image T2')
    parser.add_argument('--gt', type=str, required=True, help='Path to Ground Truth')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--patch_size', type=int, default=9, help='Patch size')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Preparation
    dm = DataManager(args.im1, args.im2, args.gt, args.patch_size)
    
    # 2. Pre-classification (Unsupervised Label Generation)
    dm.run_preclassification()
    
    # 3. Create Datasets
    x_train, y_train = dm.create_training_patches()
    train_ds = SARDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    # 4. Model Initialization
    model = CAMixer(patch_size=args.patch_size).to(device)
    
    # 5. Training
    print("Start Training CAMixer...")
    train_model(model, train_loader, args.epochs, device)
    
    # 6. Inference & Evaluation
    result_map = inference(model, dm, device)
    
    # 7. Post-processing & Metrics
    result_map_clean = postprocess(result_map * 255)
    evaluate(dm.gt, result_map_clean)
    
    # Save result
    plt.imsave('result.png', result_map_clean, cmap='gray')
    print("Result saved to result.png")

if __name__ == "__main__":
    main()