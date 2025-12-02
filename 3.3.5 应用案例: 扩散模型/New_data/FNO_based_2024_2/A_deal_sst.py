import deal_sst_util
import numpy as np
import os
from netCDF4 import Dataset
from Mask_generate import Cloud_mask, Square_mask, Strip_mask
from Noise_generate import Random_noise, Bulk_noise
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(torch.__version__)
print(torch.cuda.is_available())
print("using {} device.".format(device))

def Normalize(batch_data):
    y = torch.FloatTensor(batch_data).to(device)
    mask = torch.FloatTensor(batch_data!= 0).to(device)
    min = 0
    max = 30
    # a, b = -1, 1
    # # print("min, max", min, max)
    # k = (b-a)/ (max-min)
    #
    # # y = (((y - mean) / (max - min)).float() )
    # y = -1 + k*(y-min)

    min= y.min()
    max= y.max()
    print("min, max ", min,max )
    # mean = sum(y * mask )/ sum(mask)
    y = (y-min)/(max-min)
    y = -1 + 2* y
    print("处理之后的min, max ", y.min(),y.max() )
    y = y.cpu().numpy()
    return y


def N_S_deal(data,N_S):
    """:param signal: 原始信号:param SNR: 添加噪声的信噪比:return: 生成的噪声"""
    # signal= signal.cpu().numpy()
    SNR = 1 / N_S
    noise_all =[]
    for i in range(data.shape[0]):
        signal= data[i]
        noise=np.random.randn(*signal.shape) # *signal.shape 获取样本序列的尺寸
        noise=noise-np.mean(noise)
        signal_power=(1/(signal.shape[0]*signal.shape[1]))*np.sum(np.power(signal,2))
        noise_variance=signal_power/np.power(10,(SNR/10))
        noise=(np.sqrt(noise_variance)/np.std(noise))*noise
        noise_all.append(noise)
    data = data+noise_all
    return data

def generate_data(dataname, lat_start, lat_end, lon_start, lon_end, path, mask_type,N_S_ratio,corrup_rate):
    """
    print("********************************************************************************")
    print("处理真值数据")
    """
    # train_data_all = []
    # test_data_all = []
    #
    test_data_all = []
    train_data_all = []
    num = 0
   #  print("开始处理")
   #  file_path = '../L4_12H/'
   #
   #  ## MUL_OPER_SST_L4A_FU_12H_20220201T12_dps_050_10_sst
   # # # 1月 - 8 月： 1——6月训练， 7-8测试
   #  all_num= 0
   #  train_num = 0
   #  test_num = 0
   #  for filename in os.listdir(file_path):
   #      if os.path.exists(file_path +filename +'/nc'):
   #          count = 1
   #          for nc_name in os.listdir(file_path +filename +'/nc'):
   #              num_file =len(os.listdir(file_path +filename +'/nc'))
   #              rest_file_num = num_file % 7
   #              if num_file >=7 and count <= num_file-rest_file_num:
   #                  nc_file_path = file_path + filename + '/nc/' + nc_name
   #                  data = deal_sst_util.get_data_nc(nc_file_path, dataname, lat_start, lat_end, lon_start, lon_end)
   #                  data, N = deal_sst_util.dealNaN(data)  ##dealNaN——>将nc中的Nan写成0
   #                  print("计数{}/{},文件名为:{}".format(count, num_file, nc_file_path))
   #                  print("第{}个文件的陆地点有{}({})个".format(all_num, N,
   #                                                              (lon_end - lon_start) * (lat_end - lat_start)))
   #                  print("min,max", data.min(), data.max())
   #
   #                  if filename in ("2023_01", "2023_02","2023_03", "2023_04"):
   #                      test_data_all.append(data)
   #                      test_num = test_num + 1
   #                      all_num = all_num + 1
   #                  elif filename in ("2022_01", "2022_02", "2022_03", "2022_04", "2022_05", "2022_06","2022_07", "2022_08", "2022_09", "2022_10", "2022_11", "2022_12"):
   #                      train_data_all.append(data)
   #                      train_num = train_num + 1
   #                      all_num = all_num + 1
   #                  count = count+1
   #              else:
   #                  break
   #      else:
   #          break
   #
   #  print("文件总数量：",all_num)
   #  print("训练文件总数量：",train_num)
   #  print("测试文件总数量：",test_num)
   #
   #  deal_sst_util.cache_all("./data/" + path + "_Train_real.h5", train_data_all)
   #  print("****存储文件至{}".format('./data/' + path + '_Train_real.h5'))
   #  deal_sst_util.cache_all("./data/" + path + "_Test_real.h5", test_data_all)
   #  print("****存储文件至{}".format('./data/' + path + '_Test_real.h5'))


    train_data = deal_sst_util.read_cache_all('./data/' + path + '_Train_real.h5')
    train_data = np.array(train_data)
    test_data = deal_sst_util.read_cache_all('./data/' + path + '_Test_real.h5')
    test_data = np.array(test_data)

    #
    train_data = Normalize(train_data)
    test_data = Normalize(test_data)
    print("")

    print("********************************************************************************")
    print("为完整图像添加不同形状Mask")
    if mask_type == 'Cloud_mask':
        train_mask1,train_mask2 = Cloud_mask(train_data.shape[0], corrup_rate)  # 可选 Cloud_mask/Square_mask/Strip_mask (函数内部可以选mask的位置，默认为right)
        test_mask1,test_mask2 = Cloud_mask(test_data.shape[0], corrup_rate)  # 可选 Cloud_mask/Square_mask/Strip_mask (函数内部可以选mask的位置，默认为right)

    elif mask_type == 'Square_mask':
        mask = Square_mask(train_data.shape[0])
    elif mask_type == 'Strip_mask':
        mask = Strip_mask(train_data.shape[0])
    """
    mask min max: 0 , 1  (0为遮挡区域，1为不遮挡区域）
    mask shape :(84, 64, 64)
    mask type : numpy.ndarray
    """

    train_real_data = []
    for i in range(train_data.shape[0]):
        train_real_data.append(train_data[i])
    train_real_data = np.array(train_real_data)

    test_real_data = []
    for i in range(test_data.shape[0]):
        test_real_data.append(test_data[i])
    test_real_data = np.array(test_real_data)


    train_data = N_S_deal(train_data,N_S_ratio)
    test_data = N_S_deal(test_data,N_S_ratio)
    # print("2:type:", test_data.type)
    for i in range(train_data.shape[0]):
        if i+1 >0 and (i+1)%7==0:
            train_data[i] = train_data[i]* train_mask1[i]
        else:
            train_data[i] = train_data[i]* train_mask2[i]

    for i in range(test_data.shape[0]):
        if i+1 >0 and (i+1)%7==0:
            test_data[i] = test_data[i]* test_mask1[i]
        else:
            test_data[i] = test_data[i]* test_mask2[i]
    # train_data = train_data * train_mask1
    # test_data = test_data * test_mask1


    """
    data_ shape :(84, 64, 64)
    data_ type : numpy.ndarray
    """

    # print("********************************************************************************")
    # print("基于mask遮盖后添加不同类型噪声")
    # if noise_type == "random_noise":
    #     noise = Random_noise(noise_random_ratio, data_.shape[0], data_.shape[1], data_.shape[2])
    # elif noise_type == "bulk_noise":
    #     noise = Bulk_noise(bulk_noise_size, data_.shape[0])
    # """
    # noise shape :(84, 64, 64)
    # noise type : numpy.ndarray
    # """
    #
    # data_ = data_ * noise
    """
    data_ shape :(84, 64, 64)
    data_ type : numpy.ndarray
    """
    """
    以下三行代码可查看每个文件中0的数量以及缺失率
    """
    for i in range(train_data.shape[0]):
        zero_num = np.count_nonzero(train_data[i] == 0)
        print("训练集中，第{}个文件有{}个0,缺失率为{:.2f}% ".format(i+1, zero_num, zero_num * 100 / (train_data.shape[1] * train_data.shape[2])))

    for i in range(test_data.shape[0]):
        zero_num = np.count_nonzero(test_data[i] == 0)
        print("测试集中，第{}个文件有{}个0,缺失率为{:.2f}% ".format(i+1, zero_num, zero_num * 100 / (test_data.shape[1] * test_data.shape[2])))


    print("********************************************************************************")
    print("计算周均值")

    def zero_to_nan(d):
        array = np.array(d)
        array[array == 0] = np.NaN
        return array

    train_data1_ = train_data.reshape([train_data.shape[0], 1, train_data.shape[1] * train_data.shape[2]])
    train_data1_ = zero_to_nan(train_data1_)
    train_data1_mean_ = []

    K=14
    for i in range(train_data.shape[0]):
        if i < int(K/2)+1:
            data1_mean = np.nanmean(train_data1_[0:K], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, train_data.shape[1], train_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            train_data1_mean_.append(data1_mean)
        elif i >= int(K / 2) + 1 and (i + K - int(K / 2)) < train_data.shape[0]:
            data1_mean = np.nanmean(train_data1_[i-int(K/2):i+K-int(K/2)], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, train_data.shape[1], train_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            train_data1_mean_.append(data1_mean)
        elif (i + K - int(K / 2)) >= train_data.shape[0]:
            data1_mean = np.nanmean(train_data1_[train_data.shape[0]-K:train_data.shape[0]], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, train_data.shape[1], train_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            train_data1_mean_.append(data1_mean)

    print("train_data1_mean_：", np.array(train_data1_mean_).shape)

    train_data2_ = train_real_data.reshape([train_real_data.shape[0], 1, train_real_data.shape[1] * train_real_data.shape[2]])
    train_data2_ = zero_to_nan(train_data2_)
    train_data2_mean_ = []
    K=14
    for i in range(train_real_data.shape[0]):
        if i < int(K/2)+1:
            data2_mean = np.nanmean(train_data2_[0:K], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, train_real_data.shape[1], train_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            train_data2_mean_.append(data2_mean)
        elif i >=int(K/2)+1 and ( i+K -int(K/2)) < train_real_data.shape[0]:
            data2_mean = np.nanmean(train_data2_[i-int(K/2):i+K-int(K/2)], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, train_real_data.shape[1], train_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            train_data2_mean_.append(data2_mean)
        elif (i + K -int(K / 2)) >= train_real_data.shape[0]:
            data2_mean = np.nanmean(train_data2_[train_real_data.shape[0]-K:train_real_data.shape[0]], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, train_real_data.shape[1], train_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            train_data2_mean_.append(data2_mean)

    print("train_data2_mean_：", np.array(train_data2_mean_).shape)



    print("处理测试的日数据")

    test_data1_ = test_data.reshape([test_data.shape[0], 1, test_data.shape[1] * test_data.shape[2]])
    test_data1_ = zero_to_nan(test_data1_)
    test_data1_mean_ = []

    K=14
    for i in range(test_data.shape[0]):
        if i < int(K/2)+1:
            data1_mean = np.nanmean(test_data1_[0:K], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, test_data.shape[1], test_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            test_data1_mean_.append(data1_mean)
        elif i >= int(K / 2) + 1 and (i + K - int(K / 2)) < test_data.shape[0]:
            data1_mean = np.nanmean(test_data1_[i-int(K/2):i+K-int(K/2)], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, test_data.shape[1], test_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            test_data1_mean_.append(data1_mean)
        elif (i + K - int(K / 2)) >= test_data.shape[0]:
            data1_mean = np.nanmean(test_data1_[test_data.shape[0]-K:test_data.shape[0]], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, test_data.shape[1], test_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            test_data1_mean_.append(data1_mean)

    print("test_data1_mean_：", np.array(test_data1_mean_).shape)

    test_data2_ = test_real_data.reshape([test_real_data.shape[0], 1, test_real_data.shape[1] * test_real_data.shape[2]])
    test_data2_ = zero_to_nan(test_data2_)
    test_data2_mean_ = []
    K=14
    for i in range(test_real_data.shape[0]):
        if i < int(K/2)+1:
            data2_mean = np.nanmean(test_data2_[0:K], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, test_real_data.shape[1], test_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            test_data2_mean_.append(data2_mean)
        elif i >=int(K/2)+1 and ( i+K -int(K/2)) < test_real_data.shape[0]:
            data2_mean = np.nanmean(test_data2_[i-int(K/2):i+K-int(K/2)], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, test_real_data.shape[1], test_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            test_data2_mean_.append(data2_mean)
        elif (i + K -int(K / 2)) >= test_real_data.shape[0]:
            data2_mean = np.nanmean(test_data2_[test_real_data.shape[0]-K:test_real_data.shape[0]], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, test_real_data.shape[1], test_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            test_data2_mean_.append(data2_mean)

    print("test_data2_mean_：", np.array(test_data2_mean_).shape)

    print("********************************************************************************")
    print("处理训练的日数据")
    train_seq_all_x, train_seq_all_y = [], []
    train_data1_mean,train_data2_mean =[], []
    sw_width = 7
    train_num_week = int(len(train_data)/sw_width)
    # print("XX", train_num_week)

    for i in range(train_num_week):
        sequence_x = train_data[sw_width*i:sw_width*i+sw_width]
        sequence_y = train_real_data[sw_width*i:sw_width*i+sw_width]
        train_seq_all_x.append(sequence_x)
        train_seq_all_y.append(sequence_y)
        train_data1_mean.append(train_data1_mean_[sw_width*i+sw_width-1])
        train_data2_mean.append(train_data2_mean_[sw_width*i+sw_width-1])

    test_seq_all_x, test_seq_all_y = [], []
    test_data1_mean,test_data2_mean =[], []
    test_num_week = int(len(test_data) / sw_width)
    for i in range(test_num_week):
        sequence_x = test_data[sw_width*i:sw_width*i+sw_width]
        sequence_y = test_real_data[sw_width*i:sw_width*i+sw_width]
        test_seq_all_x.append(sequence_x)
        test_seq_all_y.append(sequence_y)
        test_data1_mean.append(test_data1_mean_[sw_width*i+sw_width-1])
        test_data2_mean.append(test_data2_mean_[sw_width*i+sw_width-1])

    train_seq_all_x= np.array(train_seq_all_x)
    train_seq_all_y= np.array(train_seq_all_y)
    train_data1_mean = np.array(train_data1_mean)
    train_data2_mean = np.array(train_data2_mean)


    test_seq_all_x= np.array(test_seq_all_x)
    test_seq_all_y= np.array(test_seq_all_y)
    test_data1_mean = np.array(test_data1_mean)
    test_data2_mean = np.array(test_data2_mean)



    """
    以下三行代码可查看每个文件中0的数量以及缺失率
    """

    total_cor_rate_mask1 = []
    total_cor_rate_mask2 = []
    total_cor_rate = []
    total_cor_rate_ = []
    for i in range(train_seq_all_x.shape[0]):
        for j in range(train_seq_all_x.shape[1]):

            cor_zero_num = np.count_nonzero(train_seq_all_x[i][j] == 0)
            total_cor_rate.append(cor_zero_num * 100 / (train_seq_all_x.shape[2] * train_seq_all_x.shape[3]))
            if j <6:
                cor_zero_num_mask2 = np.count_nonzero(train_seq_all_x[i][j] == 0)
                total_cor_rate_mask2.append(cor_zero_num_mask2 * 100 / (train_seq_all_x.shape[2] * train_seq_all_x.shape[3]))
            if j ==6:
                cor_zero_num_mask1 = np.count_nonzero(train_seq_all_x[i][j] == 0)
                total_cor_rate_mask1.append(
                    cor_zero_num_mask1 * 100 / (train_seq_all_x.shape[2] * train_seq_all_x.shape[3]))
            print("破损的第{}个文件中的第{}张图有{}个0,缺失率为{:.2f}% ".format(i + 1, j + 1, cor_zero_num,
                                                                                cor_zero_num * 100 / (
                                                                                            train_seq_all_x.shape[2] *
                                                                                            train_seq_all_x.shape[3])))

    for i in range(train_seq_all_y.shape[0]):
        for j in range(train_seq_all_y.shape[1]):
            cor_zero_num_ = np.count_nonzero(train_seq_all_y[i][j] == 0)
            total_cor_rate_.append(cor_zero_num_ * 100 / (train_seq_all_y.shape[2] * train_seq_all_y.shape[3]))
            print("真值的第{}个文件中的第{}张图有{}个0,缺失率为{:.2f}% ".format(i + 1, j + 1, cor_zero_num,
                                                                                cor_zero_num_ * 100 / (
                                                                                        train_seq_all_y.shape[2] *
                                                                                        train_seq_all_y.shape[3])))


    print("train_seq_all_x).shape",np.array(train_seq_all_x).shape)
    print("train_seq_all_y).shape",np.array(train_seq_all_y).shape)
    print("train_data1_mean).shape", np.array(train_data1_mean).shape)
    print("train_data2_mean).shape", np.array(train_data2_mean).shape)

    print("test_seq_all_x).shape",np.array(test_seq_all_x).shape)
    print("test_seq_all_y).shape",np.array(test_seq_all_y).shape)
    print("test_data1_mean).shape", np.array(test_data1_mean).shape)
    print("test_data2_mean).shape", np.array(test_data2_mean).shape)


    print("********************************************************************************")
    print("拼接日数据和均值数据")
    train_seq_all_x = np.concatenate([train_seq_all_x, train_data1_mean], axis=1)
    train_seq_all_y = np.concatenate([train_seq_all_y, train_data2_mean], axis=1)
    test_seq_all_x = np.concatenate([test_seq_all_x, test_data1_mean], axis=1)
    test_seq_all_y = np.concatenate([test_seq_all_y, test_data2_mean], axis=1)

    print("train_seq_all_x).shape",np.array(train_seq_all_x).shape)
    print("train_seq_all_y).shape",np.array(train_seq_all_y).shape)
    print("test_seq_all_x).shape",np.array(test_seq_all_x).shape)
    print("test_seq_all_y).shape",np.array(test_seq_all_y).shape)

    co_rate_mask1 = int(np.mean(total_cor_rate_mask1))
    co_rate_mask2 = int(np.mean(total_cor_rate_mask2))
    print("********************************************************************************")
    print("筛选训练/测试数据")
    x_train, y_train, _, _ = deal_sst_util.trainTestSplit(train_seq_all_x, train_seq_all_y, 1)
    _, _, x_valid, y_valid = deal_sst_util.trainTestSplit(test_seq_all_x, test_seq_all_y, 0)

    deal_sst_util.cache('./data/{}_{}_{:.0f}_train_'.format(N_S_ratio,mask_type, co_rate_mask1) + path + '_miss.h5',
                        x_train, y_train, x_valid, y_valid)
    print("****存储文件{}".format('./data/{}_{}_{:.0f}_train_'.format(N_S_ratio, mask_type, co_rate_mask1) + path + '_miss.h5'))




    mask_save = train_mask1.tolist()
    deal_sst_util.cache_all('./data/{}1_{:.0f}'.format(mask_type, co_rate_mask1) + '.h5', mask_save)
    mask_save = train_mask2.tolist()
    deal_sst_util.cache_all('./data/{}2_{:.0f}'.format(mask_type, co_rate_mask1) + '.h5', mask_save)
    # mask = deal_sst_util.read_cache_all('./data_/{}_{:.0f}'.format(mask_type,np.mean(total_cor_rate)) + '.h5')
    # mask = np.array(mask)
    # noise_save = noise.tolist()
    # deal_sst_util.cache_all('./data_/{}_{:.0f}'.format(noise_type, co_rate) + '.h5', noise_save)
    # # noise = deal_sst_util.read_cache_all('./data_/{}_{:.0f}'.format(noise_type,np.mean(total_cor_rate)) + '.h5')
    # # noise = np.array(noise)
    print("mask存储完毕")

