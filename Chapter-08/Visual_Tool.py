import torch
import os
from matplotlib import pyplot as plt
import seaborn as sns
import deal_sst_util
from A_deal_sst import generate_data
import numpy as np
import cv2
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


def create_file(N_S_ratio, mask_type,cor_rate):
    if os.path.exists('./picture/{}_{}_{}/'.format(N_S_ratio, mask_type,cor_rate)):
        print("已有该文件夹.")
    else:
        os.mkdir(r'./picture/{}_{}_{}/'.format(N_S_ratio, mask_type,cor_rate))

def visua_and_save(data, path,N_S_ratio, mask_type,cor_rate):
    create_file(N_S_ratio, mask_type,cor_rate)
    if cor_rate==8:
        mask = deal_sst_util.read_cache_all('./data/{}1_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
        mask = np.array(mask[0])
        for i in range(mask.shape[0]): #将黑变白，白变黑
            for j in range(mask.shape[1]):
                if mask[i][j] ==1:
                    mask[i][j] = 0
                elif mask[i][j] ==0:
                    mask[i][j] = 255
        cv2.imwrite("./mask1_image.png", mask)
        mask1 =mask
        mask = deal_sst_util.read_cache_all('./data/{}2_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
        mask = np.array(mask[0])
        for i in range(mask.shape[0]): #将黑变白，白变黑
            for j in range(mask.shape[1]):
                if mask[i][j] ==1:
                    mask[i][j] = 0
                elif mask[i][j] ==0:
                    mask[i][j] = 255
        cv2.imwrite("./mask2_image.png", mask)
        mask2 =mask
    #
    # noise = deal_sst_util.read_cache_all('./data/{}_{:.0f}'.format(noise_type,cor_rate) + '.h5')
    # noise = np.array(noise[0])
    # for i in range(noise.shape[0]):  # 将黑变白，白变黑
    #     for j in range(noise.shape[1]):
    #         if noise[i][j] == 1:
    #             noise[i][j] = 0
    #         elif noise[i][j] == 0:
    #             noise[i][j] = 255
    # cv2.imwrite("./noise_image.png", noise)

    # mask_image = cv2.imread("./mask_image.png")
    # noise_image = cv2.imread("./noise_image.png")
    # mask_ =  cv2.add(mask_image,noise_image)
    # cv2.imwrite("./mask_fusion.png", mask_)
    print("data.shape",data.shape)
    for i in range(data.shape[0]):
        y = torch.FloatTensor(data[i]).to(device)
        mask = torch.FloatTensor(data[i] != 0).to(device)

        # min = 271.34999999999997
        # max = 309.34999999999997
        # y = ((y-min)/(max - min ))*35
        # y=y*mask

        for j in range(y.shape[0]):
            # data2.shape(200, 200)
            if j== 7:
                filename= 'average'
            else:
                filename ='dayily'
            data2 = y[j]
            data2 = torch.squeeze(data2)
            data2 = torch.squeeze(data2).cpu().detach().numpy() #data2.shape torch.Size([200, 200])
            if path == "train_true" or path == "valid_true":
                """画图"""
                plt.figure()
                ax = sns.heatmap(data2, cmap='jet', square=False, vmin=-1,vmax=1)
                plt.xlabel("°E",fontsize =20,style = "normal",labelpad=-5.0, rotation=0, x=1.09)#fontweight ='bold',
                plt.ylabel("°N",fontsize =20,style = "normal",labelpad=-33.0, rotation=1, y=1.01) # fontweight ='bold',
                #fontsize  可选  normal/italic/oblique

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                # lat_start = 1280  # 维度起点 26N
                # lat_end = 1344  # 维度终点  23N(22.8)
                # lon_start = 5800  # 经度起点  110E
                # lon_end = 5864  # 经度终点  113E(113.2)
                name_list = ('110','110.5' ,'111', '111.5', '112', '112.5','113')
                plt.xticks(np.arange(0,65,64/6), name_list,rotation=0) #如果是np.arange(0,64,64/6),以为最后一个格的索引是63，所以最后一个刻度显示不出
                name_list = ('26', '25.5','25','24.5' ,'24', '23.5', '23')
                plt.yticks(np.arange(0, 65, 64/ 6), name_list)
                plt.savefig('./picture/{}_{}_{}/'.format(N_S_ratio,mask_type,cor_rate)+ path +'-'
                           + str(i)+ filename + str(j)+'.png', dpi=300, bbox_inches='tight')
                # plt.show()

                print('保存完成...')

            # 叠加mask， 让mask显示为白色
            elif path == "train_miss" or path =="valid_miss" :
                """画图"""
                if j == 6:
                    mask = cv2.imread("./mask1_image.png")
                elif j<6:
                    mask = cv2.imread("./mask2_image.png")
                mask_ = cv2.resize(mask, dsize=(data2.shape[0],data2.shape[1]), dst=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                mask_ = mask_[:,:,0]
                mask_ = mask_>220
                plt.figure()
                ax = sns.heatmap(data2, cmap='jet', square=False, vmin=-1, vmax=1)#mask=mask_,
                plt.xlabel("°E", fontsize=20, style="normal", labelpad=-5.0, rotation=0, x=1.09)  # fontweight ='bold',
                plt.ylabel("°N", fontsize=20, style="normal", labelpad=-33.0, rotation=1, y=1.01)  # fontweight ='bold',
                # fontsize  可选  normal/italic/oblique

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                name_list = ('110','110.5' ,'111', '111.5', '112', '112.5','113')
                plt.xticks(np.arange(0, 65, 64 / 6), name_list,
                           rotation=0)  # 如果是np.arange(0,64,64/6),以为最后一个格的索引是63，所以最后一个刻度显示不出
                name_list = ('26', '25.5','25','24.5' ,'24', '23.5', '23')
                plt.yticks(np.arange(0, 65, 64 / 6), name_list)
                """
                lat_start = 2877  # 维度起点 -29.875 S
                lat_end = 2941  # 维度终点  -32.5416 S
                lon_start = 2517  # 经度起点  -75.125 W
                lon_end = 2581  # 经度终点  -72.458 W
                """
                plt.savefig('./picture/{}_{}_{}/'.format(N_S_ratio,mask_type,cor_rate)+ path +'-'
                           + str(i)+ filename + str(j)+'.png', dpi=300, bbox_inches='tight')
                # plt.show()
                print("OK",path)





"""
# lon -75.69195281054697W~
# lat -30.016255804375934S~
南美洲左侧
"""
dataname = 'SST'
lat_start = 1280  # 维度起点 26N
lat_end = 1344  # 维度终点  23N(22.8)
lon_start = 5800  # 经度起点  110E
lon_end = 5864  # 经度终点  113E

save_file = "South_Sea"  #保存的文件的地点名字
mask_type = "Cloud_mask"  # 可选 Cloud_mask/Square_mask/Strip_mask (函数内部可以选mask的位置，默认为right)
# noise_type = "random_noise"   # 可选 bulk_noise/ random_noise
# noise_random_ratio = 0.06    # noise_type = "random_noise"  要选择随机噪声中0的概率
# bulk_noise_size = "bulk_small"    # noise_type = "bulk_noise"  bulk噪声的尺寸 可选  bulk_small/bulk_meso/bulk_big
N_S_ratio = 0.1
corrup_rate = 8  #读取文件的名字中的缺失率

#step1
# generate_data(dataname, lat_start,lat_end,lon_start,lon_end,save_file,mask_type,N_S_ratio,corrup_rate)
#
x_train, y_train, x_valid, y_valid = deal_sst_util.read_cache('./data/{}_{}_{}_train_'.format(N_S_ratio, mask_type, corrup_rate )+save_file+'_miss.h5')
# print(np.array(x_train).shape, x_valid.shape)
#           #step2 #要可视化的数据， 读取的文件的地点名称, 保存图片的前缀
# visua_and_save(x_train, 'train_miss',N_S_ratio, mask_type,corrup_rate)
visua_and_save(y_train, 'train_true',N_S_ratio, mask_type,corrup_rate)
# visua_and_save(x_valid, 'valid_miss',N_S_ratio, mask_type,corrup_rate)
visua_and_save(y_valid, 'valid_true',N_S_ratio, mask_type,corrup_rate)
# print("可视化结束")