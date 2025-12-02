import numpy as np
import math
from skimage import measure

def image_normalize(data):
    """Data normalization [cite: 342-346]"""
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)

def image_padding(data, r):
    if len(data.shape) == 3:
        return np.pad(data, ((r, r), (r, r), (0, 0)), 'constant', constant_values=0)
    if len(data.shape) == 2:
        return np.pad(data, r, 'constant', constant_values=0)

def postprocess(res):
    """Post-processing to remove small noise regions"""
    res_new = res.copy()
    res_label = measure.label(res, connectivity=2)
    num = res_label.max()
    for i in range(1, num + 1):
        idy, idx = np.where(res_label == i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0  # Remove small connected components
    return res_new

def evaluate(gtImg, tstImg):
    """
    Calculate FP, FN, OE, PCC, Kappa [cite: 184]
    """
    # Ensure binary format (0 and 255)
    gt = gtImg.copy()
    tst = tstImg.copy()
    gt[gt > 128] = 255
    gt[gt < 128] = 0
    tst[tst > 128] = 255
    tst[tst < 128] = 0

    ylen, xlen = gt.shape
    
    # Calculate counts
    # TP: Changed detected as Changed (Both 255)
    # TN: Unchanged detected as Unchanged (Both 0)
    # FP: Unchanged detected as Changed (gt=0, tst=255)
    # FN: Changed detected as Unchanged (gt=255, tst=0)
    
    FA = np.sum((gt == 0) & (tst == 255)) # False Alarm (FP)
    MA = np.sum((gt == 255) & (tst == 0)) # Missed Alarm (FN)
    
    label_0 = np.sum(gt == 0)   # Total negatives
    label_1 = np.sum(gt == 255) # Total positives
    
    OE = FA + MA
    PCC = 1 - OE / (ylen * xlen)
    
    PRE = ((label_1 + FA - MA) * label_1 + (label_0 + MA - FA) * label_0) / ((ylen * xlen) ** 2)
    KC = (PCC - PRE) / (1 - PRE)
    
    print('=== Change Detection Metrics ===')
    print(f'False Positives (FP): {FA}')
    print(f'False Negatives (FN): {MA}')
    print(f'Overall Error   (OE): {OE}')
    print(f'PCC Accuracy    (%): {PCC*100:.2f}')
    print(f'Kappa Coeff     (%): {KC*100:.2f}')
    return PCC, KC