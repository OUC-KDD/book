import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from .utils import image_padding, image_normalize
# 注意：preclassify 需要从外部文件导入，确保目录下有 preclassify.py
try:
    from .preclassify import dicomp, hcluster
except ImportError:
    print("Warning: 'preclassify.py' not found. Ensure it is in the python path.")

class SARDataset(Dataset):
    def __init__(self, data, labels):
        self.x_data = torch.FloatTensor(data)
        self.y_data = torch.LongTensor(labels)
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class DataManager:
    def __init__(self, im1_path, im2_path, gt_path, patch_size=9):
        self.im1 = io.imread(im1_path)
        if self.im1.ndim == 3:
            self.im1 = self.im1[:, :, 0]
        self.im1 = self.im1.astype(np.float32)

        self.im2 = io.imread(im2_path)
        if self.im2.ndim == 3:
            self.im2 = self.im2[:, :, 0]
        self.im2 = self.im2.astype(np.float32)

        self.gt = io.imread(gt_path)
        if self.gt.ndim == 3:
            self.gt = self.gt[:, :, 0]
        self.gt = self.gt.astype(np.float32)

        self.patch_size = patch_size
        self.h, self.w = self.im1.shape
        self.preclassify_lab = None
        self.mdata = None

    def run_preclassification(self):
        """
        Generate pseudo-labels using Hierarchical FCM [cite: 100-101]
        """
        print("Running pre-classification (this may take a while)...")
        im_di = dicomp(self.im1, self.im2) # Log-ratio difference
        pix_vec = im_di.reshape([self.h * self.w, 1])
        
        # 1=Unchanged, 2=Changed, 1.5=Uncertain
        self.preclassify_lab = hcluster(pix_vec, im_di) 
        print("Pre-classification finished.")
        
        # Prepare 3-channel input: [Image1, Image2, DifferenceMap]
        self.mdata = np.zeros([self.h, self.w, 3], dtype=np.float32)
        self.mdata[:,:,0] = self.im1
        self.mdata[:,:,1] = self.im2
        self.mdata[:,:,2] = im_di
        
        return self.preclassify_lab

    def create_training_patches(self, num_samples=10000):
        """
        Extract patches for pixels with confident labels (1 or 2) [cite: 102]
        """
        if self.preclassify_lab is None:
            raise ValueError("Run preclassification first.")
            
        labels = self.preclassify_lab
        # Padding
        margin = int((self.patch_size - 1) / 2)
        padded_data = image_padding(self.mdata, margin)
        
        # Find confident pixels
        idx_unchanged = np.where(labels == 1) # Label 1 -> 0
        idx_changed = np.where(labels == 2)   # Label 2 -> 1
        
        # Extract patches logic (simplified for readability)
        # In a real scenario, efficient vectorization is better, 
        # here we follow the logic of the original nb for fidelity.
        
        # ... (Patch extraction logic similar to notebook) ...
        # For brevity, assuming helper function or implementation:
        
        X_patches = []
        y_labels = []
        
        # Collect all valid patches
        for r, c in zip(idx_unchanged[0], idx_unchanged[1]):
             patch = padded_data[r:r+self.patch_size, c:c+self.patch_size]
             X_patches.append(patch)
             y_labels.append(0) # 0 for unchanged
             
        for r, c in zip(idx_changed[0], idx_changed[1]):
             patch = padded_data[r:r+self.patch_size, c:c+self.patch_size]
             X_patches.append(patch)
             y_labels.append(1) # 1 for changed

        X_patches = np.array(X_patches)
        y_labels = np.array(y_labels)
        
        # Shuffle and select subset to balance/limit size
        indices = np.arange(len(y_labels))
        np.random.shuffle(indices)
        indices = indices[:num_samples] # Limit training samples
        
        X_train = X_patches[indices].transpose(0, 3, 1, 2) # [N, C, H, W]
        y_train = y_labels[indices]
        
        return X_train, y_train

    def create_testing_patches(self):
        """Prepare all pixels for inference"""
        margin = int((self.patch_size - 1) / 2)
        padded_data = image_padding(self.mdata, margin)
        
        patches = []
        for r in range(margin, padded_data.shape[0] - margin):
            for c in range(margin, padded_data.shape[1] - margin):
                patch = padded_data[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patches.append(patch)
        
        patches = np.array(patches)
        return patches.transpose(0, 3, 1, 2)