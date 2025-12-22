#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import netCDF4
import deal_sst_util
import DINCAE
import os
import numpy as np
from mask_obtain import mask_obtain
# from pytorch_ssim import pytorch_ssim
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# # filename = "/path/to/file.nc"
# # filename = "input_file_python.nc"
# print(torch.__version__)
# print(torch.cuda.is_available())
# print("using {} device.".format(device))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(torch.__version__)
print(torch.cuda.is_available())
print("using {} device.".format(device))
# filename = '/home/haida_niejie/work/wqc/1982_real/NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA07_G_1982001-v02.0-fv01.0.nc'
# filename = 'H1C_OPER_OCT_L3A_20220804d_SST_4KM_13.h5'
# filename = 'Cloud_mask_random_noise_19_train_Pacific_miss.h5'
# filename = "oisst-avhrr-v02r01.19810901.nc"



# N_S_ratio = 0.2
# N_S_ratio = 0.3

def run(mask_type,corrup_rate,N_S_ratio):
    if corrup_rate == 8:
        mask_path = "531"  # 8%
    if corrup_rate == 25:
        mask_path = "526"  # 25%
    if corrup_rate == 46:
        mask_path = "732"  # 46%
    if corrup_rate == 68:
        mask_path = "455"  # 68%
    # varname = "SST"
    # varname = "sea_surface_temperature"
    varname = "SST"
    # varname = "sst"
    # outdir = "create_output_file.py"
    outdir = "output_file_0801"

    filenamelist = []
    # for year in range(1982, 1983):
    #     # file_path = '../1982_real/'
    #     file_path = '/home/haida_niejie/work/wqc/1982_real/'
    #     i = 1
    #     for filename in os.listdir(file_path):
    #         if i < 10:
    #             sti = '00' + str(i)
    #         else:
    #             sti = '0' + str(i)
    #         file_path2 = file_path + 'NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA07_G_1982' + sti + '-v02.0-fv01.0.nc'
    #
    #         i = i + 1
    #
    #
    #         filenamelist.append(file_path2)
    #         # if i == 2:
    #         #     break
    #
    # # print(filenamelist)
    # # DINCAE.reconstruct_gridded_nc(filename,varname,outdir)
    # DINCAE.reconstruct_gridded_nc(filenamelist,varname,outdir)
    dataname = 'SST'
    lat_start = 1280  # 维度起点 26N
    lat_end = 1344  # 维度终点  23N(22.8)
    lon_start = 5800  # 经度起点  110E
    lon_end = 5864  # 经度终点  113E(113.2)

    test_data_all = []
    train_data_all = []
    file_path = './New_data/L4_12H/'
    path = "South_Sea"
    all_num= 0
    train_num = 0
    test_num = 0
    train_time = []
    test_time = []
    # for filename in os.listdir(file_path):
    #     if os.path.exists(file_path +filename +'/nc'):
    #         count = 1
    #         for nc_name in os.listdir(file_path +filename +'/nc'):
    #             num_file =len(os.listdir(file_path +filename +'/nc'))
    #             rest_file_num = num_file % 7
    #             if num_file >=7 and count <= num_file-rest_file_num:
    #                 nc_file_path = file_path + filename + '/nc/' + nc_name
    #                 data = deal_sst_util.get_data_nc(nc_file_path, dataname, lat_start, lat_end, lon_start, lon_end)
    #                 data, N = deal_sst_util.dealNaN(data)  ##dealNaN——>将nc中的Nan写成0
    #                 print("计数{}/{},文件名为:{}".format(count, num_file, nc_file_path))
    #                 print("第{}个文件的陆地点有{}({})个".format(all_num, N,
    #                                                             (lon_end - lon_start) * (lat_end - lat_start)))
    #                 print("min,max", data.min(), data.max())
    #
    #                 if filename in ("2023_01", "2023_02","2023_03", "2023_04"):
    #                     test_data_all.append(data_)
    #                     test_time.append(test_num)
    #                     test_num = test_num + 1
    #                     all_num = all_num + 1
    #                 elif filename in ("2022_01", "2022_02", "2022_03", "2022_04", "2022_05", "2022_06","2022_07", "2022_08", "2022_09", "2022_10", "2022_11", "2022_12"):
    #                     train_data_all.append(data_)
    #                     train_time.append(train_num)
    #                     train_num = train_num + 1
    #                     all_num = all_num + 1
    #                 count = count+1
    #             else:
    #                 break
    #     else:
    #         break
    # deal_sst_util.cache_all("./data/" + path + "_Train_real.h5", train_data_all)
    # print("****存储文件至{}".format('./data/' + path + '_Train_real.h5'))
    # deal_sst_util.cache_all("./data/" + path + "_Test_real.h5", test_data_all)
    # print("****存储文件至{}".format('./data/' + path + '_Test_real.h5'))
    #
    # deal_sst_util.cache_all("./data/" + path + "_Train_time.h5", train_time)
    # print("****存储文件至{}".format('./data/' + path + '_Train_time.h5'))
    # deal_sst_util.cache_all("./data/" + path + "_Test_time.h5", test_time)
    # print("****存储文件至{}".format('./data/' + path + '_Test_time.h5'))

    train_data = deal_sst_util.read_cache_all('./data/' + path + '_Train_real.h5')
    train_data = np.array(train_data)
    test_data = deal_sst_util.read_cache_all('./data/' + path + '_Test_real.h5')
    test_data = np.array(test_data)

    train_time = deal_sst_util.read_cache_all('./data/' + path + '_Train_time.h5')
    train_time = np.array(train_time)
    test_time = deal_sst_util.read_cache_all('./data/' + path + '_Test_time.h5')
    test_time = np.array(test_time)

    print("train_data.shape",train_data.shape)
    print("test_data.shape",test_data.shape)
    print("train_time.shape",train_time.shape)
    print("test_time.shape",test_time.shape)
    train_real_data = []
    for i in range(train_data.shape[0]):
        train_real_data.append(train_data[i])
    train_real_data = np.array(train_real_data)

    test_real_data = []
    for i in range(test_data.shape[0]):
        test_real_data.append(test_data[i])
    test_real_data = np.array(test_real_data)


    for i in range(train_data.shape[0]):
        zero_num = np.count_nonzero(train_data[i] == 0)
        print("训练集中，第{}个文件有{}个0,缺失率为{:.2f}% ".format(i + 1, zero_num, zero_num * 100 / (
                    train_data.shape[1] * train_data.shape[2])))

    for i in range(test_data.shape[0]):
        zero_num = np.count_nonzero(test_data[i] == 0)
        print("测试集中，第{}个文件有{}个0,缺失率为{:.2f}% ".format(i + 1, zero_num, zero_num * 100 / (
                    test_data.shape[1] * test_data.shape[2])))

    print("********************************************************************************")
    print("计算周均值")


    def zero_to_nan(d):
        array = np.array(d)
        array[array == 0] = np.nan
        return array


    train_data1_ = train_data.reshape([train_data.shape[0], 1, train_data.shape[1] * train_data.shape[2]])
    train_data1_ = zero_to_nan(train_data1_)
    train_data1_mean_ = []

    K = 14
    for i in range(train_data.shape[0]):
        if i < int(K / 2) + 1:
            data1_mean = np.nanmean(train_data1_[0:K], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, train_data.shape[1], train_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            train_data1_mean_.append(data1_mean)
        elif i >= int(K / 2) + 1 and (i + K - int(K / 2)) < train_data.shape[0]:
            data1_mean = np.nanmean(train_data1_[i - int(K / 2):i + K - int(K / 2)], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, train_data.shape[1], train_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            train_data1_mean_.append(data1_mean)
        elif (i + K - int(K / 2)) >= train_data.shape[0]:
            data1_mean = np.nanmean(train_data1_[train_data.shape[0] - K:train_data.shape[0]], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, train_data.shape[1], train_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            train_data1_mean_.append(data1_mean)

    print("train_data1_mean_：", np.array(train_data1_mean_).shape)

    train_data2_ = train_real_data.reshape(
        [train_real_data.shape[0], 1, train_real_data.shape[1] * train_real_data.shape[2]])
    train_data2_ = zero_to_nan(train_data2_)
    train_data2_mean_ = []
    K = 14
    for i in range(train_real_data.shape[0]):
        if i < int(K / 2) + 1:
            data2_mean = np.nanmean(train_data2_[0:K], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, train_real_data.shape[1], train_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            train_data2_mean_.append(data2_mean)
        elif i >= int(K / 2) + 1 and (i + K - int(K / 2)) < train_real_data.shape[0]:
            data2_mean = np.nanmean(train_data2_[i - int(K / 2):i + K - int(K / 2)], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, train_real_data.shape[1], train_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            train_data2_mean_.append(data2_mean)
        elif (i + K - int(K / 2)) >= train_real_data.shape[0]:
            data2_mean = np.nanmean(train_data2_[train_real_data.shape[0] - K:train_real_data.shape[0]], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, train_real_data.shape[1], train_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            train_data2_mean_.append(data2_mean)

    print("train_data2_mean_：", np.array(train_data2_mean_).shape)

    print("处理测试的日数据")

    test_data1_ = test_data.reshape([test_data.shape[0], 1, test_data.shape[1] * test_data.shape[2]])
    test_data1_ = zero_to_nan(test_data1_)
    test_data1_mean_ = []

    K = 14
    for i in range(test_data.shape[0]):
        if i < int(K / 2) + 1:
            data1_mean = np.nanmean(test_data1_[0:K], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, test_data.shape[1], test_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            test_data1_mean_.append(data1_mean)
        elif i >= int(K / 2) + 1 and (i + K - int(K / 2)) < test_data.shape[0]:
            data1_mean = np.nanmean(test_data1_[i - int(K / 2):i + K - int(K / 2)], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, test_data.shape[1], test_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            test_data1_mean_.append(data1_mean)
        elif (i + K - int(K / 2)) >= test_data.shape[0]:
            data1_mean = np.nanmean(test_data1_[test_data.shape[0] - K:test_data.shape[0]], axis=0).round(2)
            data1_mean = data1_mean.reshape([1, test_data.shape[1], test_data.shape[2]])
            data1_mean = np.nan_to_num(data1_mean)
            test_data1_mean_.append(data1_mean)

    print("test_data1_mean_：", np.array(test_data1_mean_).shape)

    test_data2_ = test_real_data.reshape([test_real_data.shape[0], 1, test_real_data.shape[1] * test_real_data.shape[2]])
    test_data2_ = zero_to_nan(test_data2_)
    test_data2_mean_ = []
    K = 14
    for i in range(test_real_data.shape[0]):
        if i < int(K / 2) + 1:
            data2_mean = np.nanmean(test_data2_[0:K], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, test_real_data.shape[1], test_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            test_data2_mean_.append(data2_mean)
        elif i >= int(K / 2) + 1 and (i + K - int(K / 2)) < test_real_data.shape[0]:
            data2_mean = np.nanmean(test_data2_[i - int(K / 2):i + K - int(K / 2)], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, test_real_data.shape[1], test_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            test_data2_mean_.append(data2_mean)
        elif (i + K - int(K / 2)) >= test_real_data.shape[0]:
            data2_mean = np.nanmean(test_data2_[test_real_data.shape[0] - K:test_real_data.shape[0]], axis=0).round(2)
            data2_mean = data2_mean.reshape([1, test_real_data.shape[1], test_real_data.shape[2]])
            data2_mean = np.nan_to_num(data2_mean)
            test_data2_mean_.append(data2_mean)

    print("test_data2_mean_：", np.array(test_data2_mean_).shape)

    print("********************************************************************************")
    print("处理训练的日数据")
    train_seq_all_x, train_seq_all_y = [], []
    train_data1_mean, train_data2_mean = [], []
    sw_width = 7
    train_num_week = int(len(train_data) / sw_width)
    # print("XX", train_num_week)
    train_all_time =[]
    for i in range(train_num_week):
        sequence_x = train_data[sw_width * i:sw_width * i + sw_width]
        sequence_y = train_real_data[sw_width * i:sw_width * i + sw_width]
        sequence_time = train_time[sw_width * i:sw_width * i + sw_width]
        train_seq_all_x.append(sequence_x)
        train_seq_all_y.append(sequence_y)

        train_all_time.append(sequence_time)

        train_data1_mean.append(train_data1_mean_[sw_width * i + sw_width - 1])
        train_data2_mean.append(train_data2_mean_[sw_width * i + sw_width - 1])

    test_seq_all_x, test_seq_all_y = [], []
    test_data1_mean, test_data2_mean = [], []
    test_num_week = int(len(test_data) / sw_width)
    test_all_time=[]
    for i in range(test_num_week):
        sequence_x = test_data[sw_width * i:sw_width * i + sw_width]
        sequence_y = test_real_data[sw_width * i:sw_width * i + sw_width]
        sequence_time = test_time[sw_width * i:sw_width * i + sw_width]
        test_seq_all_x.append(sequence_x)
        test_seq_all_y.append(sequence_y)
        test_all_time.append(sequence_time)
        test_data1_mean.append(test_data1_mean_[sw_width * i + sw_width - 1])
        test_data2_mean.append(test_data2_mean_[sw_width * i + sw_width - 1])

    train_seq_all_x = np.array(train_seq_all_x)
    train_seq_all_y = np.array(train_seq_all_y)
    train_all_time = np.array(train_all_time)
    test_all_time = np.array(test_all_time)


    train_data1_mean = np.array(train_data1_mean)
    train_data2_mean = np.array(train_data2_mean)

    test_seq_all_x = np.array(test_seq_all_x)
    test_seq_all_y = np.array(test_seq_all_y)
    test_data1_mean = np.array(test_data1_mean)
    test_data2_mean = np.array(test_data2_mean)

    """
    以下三行代码可查看每个文件中0的数量以及缺失率
    """

    total_cor_rate = []
    total_cor_rate_ = []
    for i in range(train_seq_all_x.shape[0]):
        for j in range(train_seq_all_x.shape[1]):
            cor_zero_num = np.count_nonzero(train_seq_all_x[i][j] == 0)
            total_cor_rate.append(cor_zero_num * 100 / (train_seq_all_x.shape[2] * train_seq_all_x.shape[3]))
            print("破损的第{}个文件中的第{}张图有{}个0,缺失率为{:.2f}% ".format(i + 1, j + 1, cor_zero_num,
                                                                                cor_zero_num * 100 / (
                                                                                        train_seq_all_x.shape[2] *
                                                                                        train_seq_all_x.shape[3])))

    for i in range(train_seq_all_y.shape[0]):
        for j in range(train_seq_all_y.shape[1]):
            cor_zero_num_ = np.count_nonzero(train_seq_all_y[i][j] == 0)
            total_cor_rate_.append(cor_zero_num_ * 100 / (train_seq_all_y.shape[2] * train_seq_all_y.shape[3]))
            print("破损的第{}个文件中的第{}张图有{}个0,缺失率为{:.2f}% ".format(i + 1, j + 1, cor_zero_num,
                                                                                cor_zero_num_ * 100 / (
                                                                                        train_seq_all_y.shape[2] *
                                                                                        train_seq_all_y.shape[3])))

    print("train_seq_all_x).shape", np.array(train_seq_all_x).shape)
    print("train_seq_all_y).shape", np.array(train_seq_all_y).shape)
    print("train_data1_mean).shape", np.array(train_data1_mean).shape)
    print("train_data2_mean).shape", np.array(train_data2_mean).shape)

    print("test_seq_all_x).shape", np.array(test_seq_all_x).shape)
    print("test_seq_all_y).shape", np.array(test_seq_all_y).shape)
    print("test_data1_mean).shape", np.array(test_data1_mean).shape)
    print("test_data2_mean).shape", np.array(test_data2_mean).shape)

    print("********************************************************************************")
    print("拼接日数据和均值数据")
    # train_seq_all_x = np.concatenate([train_seq_all_x, train_data1_mean], axis=1)
    # train_seq_all_y = np.concatenate([train_seq_all_y, train_data2_mean], axis=1)
    # test_seq_all_x = np.concatenate([test_seq_all_x, test_data1_mean], axis=1)
    # test_seq_all_y = np.concatenate([test_seq_all_y, test_data2_mean], axis=1)

    print("train_seq_all_x).shape", np.array(train_seq_all_x).shape)
    print("train_seq_all_y).shape", np.array(train_seq_all_y).shape)
    print("test_seq_all_x).shape", np.array(test_seq_all_x).shape)
    print("test_seq_all_y).shape", np.array(test_seq_all_y).shape)


    ######生成数据
    train_data_all = train_seq_all_y[:, 6]
    test_data_all = test_seq_all_y[:, 6]
    train_time_all  = train_all_time[:,6]
    test_time_all  = test_all_time[:,6]
    print("train_data_all.shape",train_data_all.shape)
    print("test_data_all.shape",test_data_all.shape)
    """
        dayofyear = np.array([d.timetuple().tm_yday for d in time])
    """
    print("train_time_all.shape",train_time_all.shape)
    print("test_time_all.shape",test_time_all.shape)
    lon_all = np.array([61.687496,61.72916,61.77083,61.812496 ,61.85416 , 61.89583 , 61.937496,
      61.97916,  62.02083 , 62.062496 ,62.10416  ,62.14583,  62.187496 ,62.22916,
      62.27083 , 62.312496, 62.35416 , 62.39583  ,62.437496 ,62.47916 , 62.52083,
      62.562496 ,62.60416 , 62.64583 , 62.687496, 62.72916,  62.77083 , 62.812496,
      62.85416 , 62.89583 , 62.937496, 62.97916,  63.02083 , 63.062496, 63.10416,
      63.14583  ,63.187496 ,63.22916 , 63.27083 , 63.312496, 63.35416,  63.39583,
      63.437496, 63.47916 , 63.52083 , 63.562496, 63.60416 , 63.64583 , 63.687496,
      63.72916 , 63.77083,  63.812496, 63.85416 , 63.89583 , 63.937496, 63.97916,
      64.02083 , 64.06249 , 64.104164 ,64.14583,  64.18749,  64.229164, 64.27083,
      64.31249 ])
    lat_all= np.array([36.645836 ,36.604168, 36.562504 ,36.520836 ,36.479168, 36.437504, 36.395836,
      36.354168, 36.312504, 36.270836 ,36.229168 ,36.187504 ,36.145836 ,36.104168,
      36.062504 ,36.020836 ,35.979168 ,35.937504 ,35.895836 ,35.854168 ,35.812504,
      35.770836 ,35.729168, 35.687504, 35.645836, 35.604168 ,35.562504, 35.520836,
      35.479168, 35.437504 ,35.395836 ,35.354168 ,35.312504, 35.270836, 35.229168,
      35.187504, 35.145836 ,35.104168, 35.062504 ,35.020836 ,34.979168, 34.937504,
      34.895836, 34.854168 ,34.812504 ,34.770836 ,34.729168, 34.687504, 34.645836,
      34.604168, 34.562504, 34.520836 ,34.479168 ,34.437504, 34.395836 ,34.354168,
      34.312504, 34.270836 ,34.229168,34.187504, 34.145836, 34.104168 ,34.062504,
      34.020836])

    train_lon_all, train_lat_all = [],[]
    for i in range(train_data_all.shape[0]):
        train_lon_all.append(lon_all)
        train_lat_all.append(lat_all)
    train_lon_all = np.array(train_lon_all)
    train_lat_all = np.array(train_lat_all)

    test_lon_all, test_lat_all = [],[]
    for i in range(test_data_all.shape[0]):
        test_lon_all.append(lon_all)
        test_lat_all.append(lat_all)
    test_lon_all = np.array(test_lon_all)
    test_lat_all = np.array(test_lat_all)

    print("train_lon_all.shape",train_lon_all.shape)
    print("train_lat_all.shape",train_lat_all.shape)
    print("test_lon_all.shape",test_lon_all.shape)
    print("test_lat_all.shape",test_lat_all.shape)

    missing = 1-mask_obtain("mask", mask_type, corrup_rate, mask_num=0)
    # missing_all= np.array(missing, np.bool)
    missing_all= np.array(missing,np.float64)
    print("missing_all",missing_all)
    # missing_all= np.zeros((64,64), dtype=bool)

    train_missing_all= []
    for i in range(train_data_all.shape[0]):
        train_missing_all.append(missing_all)
    train_missing_all = np.array(train_missing_all)

    test_missing_all= []
    for i in range(test_data_all.shape[0]):
        test_missing_all.append(missing_all)
    test_missing_all = np.array(test_missing_all)
    print("train_missing_all.shape",train_missing_all.shape)
    print("test_missing_all.shape",test_missing_all.shape)

    print("test_missing_all", test_missing_all)
    print("train_missing_all", train_missing_all)

    # mask = mask_obtain("mask", mask_type, corrup_rate, mask_num=0)
    # mask_all= np.array(mask,np.bool)

    # mask_all= np.ones((64,64), dtype=bool)
    mask_all= np.ones((64,64))
    print('mask_all', mask_all)
    train_mask_all= []
    for i in range(train_data_all.shape[0]):
        train_mask_all.append(mask_all)
    train_mask_all = np.array(train_mask_all)

    test_mask_all= []
    for i in range(test_data_all.shape[0]):
        test_mask_all.append(mask_all)
    test_mask_all = np.array(test_mask_all)
    print("train_mask_all", train_mask_all)
    print("test_mask_all", test_mask_all)

    print("train_mask_all.shape",train_mask_all.shape)
    print("test_mask_all.shape",test_mask_all.shape)

    # train_missing_all, test_missing_all = np.array([[]])torch.full_like

    #
    # train_data_all, train_missing_all, train_mask_all,train_lon_all, train_lat_all, train_time_all=
    # test_data_all,  test_missing_all, test_mask_all,test_lon_all, test_lat_all, test_time_all  =
    #


    DINCAE.reconstruct_gridded_nc( outdir,train_data_all,
                                   train_missing_all, train_mask_all,train_lon_all, train_lat_all, train_time_all,
                                test_data_all,  test_missing_all, test_mask_all,test_lon_all, test_lat_all, test_time_all, mask_path,N_S_ratio,corrup_rate )

if __name__ == '__main__':
    mask_type = "Cloud_mask"
    for N_S_ratio in {0.3}: # 0.1 0.2 0.3
        for corrup_rate in {8}: # 68 46 25 8
            run(mask_type=mask_type,corrup_rate=corrup_rate,N_S_ratio=N_S_ratio)