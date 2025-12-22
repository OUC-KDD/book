import torch
import deal_sst_util
import numpy as np


def mask_obtain(type, mask_type, cor_rate,mask_num):
    if type== "mask":
        mask = deal_sst_util.read_cache_all('../New_data/FNO_based_2024_2/data/{}1_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
        mask = np.array(mask[0])

        if mask_num==1:
            return torch.unsqueeze(torch.unsqueeze(torch.tensor(mask).float(), dim=0),dim=0)
        elif mask_num==0:
            return mask
        elif mask_num >1:
            data = mask
            all_data = []
            for i in range(mask_num):
                all_data.append(data)
            mask_ = np.array(all_data)

            return torch.unsqueeze(torch.tensor(mask_).float(), dim=1)
    # elif type == "noise":
    #     mask = deal_sst_util.read_cache_all(
    #         '../data_/GCSC_IM/data_/{}_{:.0f}'.format(noise_type, cor_rate) + '.h5')  # mask 84 64 64
    #     mask = np.array(mask[0])
    #
    #     if mask_num == 1:
    #         return torch.unsqueeze(torch.unsqueeze(torch.tensor(mask).float(), dim=0), dim=0)
    #     else:
    #         data_ = mask
    #         all_data = []
    #         for i in range(mask_num):
    #             all_data.append(data_)
    #         mask_ = np.array(all_data)
    #
    #         return torch.unsqueeze(torch.tensor(mask_).float(), dim=1)
    #
    #
    # elif type == "fusion_mask":
    #     mask = deal_sst_util.read_cache_all(
    #         '../data_/AIN/data_/{}_{:.0f}'.format(noise_type, cor_rate) + '.h5')  # mask 84 64 64
    #     mask = np.array(mask[0])
    #     noise = deal_sst_util.read_cache_all('../data_/GCSC_IM/data_/{}_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
    #     noise = np.array(noise[0])
    #
    #     fusion_mask  = mask + noise
    #     mask = fusion_mask > 1
    #     if mask_num == 1:
    #         return torch.unsqueeze(torch.unsqueeze(torch.tensor(mask).float(), dim=0), dim=0)
    #     else:
    #         data_ = mask
    #         all_data = []
    #         for i in range(mask_num):
    #             all_data.append(data_)
    #         mask_ = np.array(all_data)
    #
    #         return torch.unsqueeze(torch.tensor(mask_).float(), dim=0)