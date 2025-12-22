#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# DINCAE: Data-Interpolating Convolutional Auto-Encoder
# Copyright (C) 2019 Alexander Barth
#
# This file is part of DINCAE.

# DINCAE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

# DINCAE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DINCAE. If not, see <http://www.gnu.org/licenses/>.

"""
DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to
reconstruct missing data in satellite observations.

For most application it is sufficient to call the function
`DINCAE.reconstruct_gridded_nc` directly.

The code is available at:
[https://github.com/gher-ulg/DINCAE](https://github.com/gher-ulg/DINCAE)
"""
import os
import random
from argparse import ArgumentError
from math import ceil, floor
from netCDF4 import Dataset, num2date  #nc文件读取
#NetCDF网络通用数据格式是一种面向数组型并适用于网络共享的数据的描述和编码标准
import numpy as np
import tensorflow as tf
from datetime import datetime       #看出是逐时数据
from test_mask import clou_mask,Normalize
#创建一个日志文件，以便更好地了解应用程序的控制流程
#目的是记录函数的名称以及我向日志输出发送信息的行号
from reconstru_visual import visua_and_save
import logging
tf.compat.v1.disable_eager_execution()

logger = logging.getLogger('root')
#日志对象，logging模块中最基础的对象，用logging.getLogger(name)方法进行初始化，name可以不填。
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

#__all__在一个函数包中可指定哪些单个的py文件能够被导入
#在更高一级的包中（一个文件夹中有多个py文件）会有一个__init__.py的文件，在这个文件书写__all__可以指定那些py文件可以被使用。
#当然当__all__没有指定内容的话，就默认全部可以使用。
__all__ = ["reconstruct","load_gridded_nc","data_generator","reconstruct_gridded_nc"]




def identity(x):
    return x

def load_gridded_nc(fname,varname, minfrac = 0.05):
    """
Load the variable `varname` from the NetCDF file `fname`. The variable `lon` is
the longitude in degrees east, `lat` is the latitude in degrees North, `time` is
a numpy datetime vector, `data_full` is a 3-d array with the data_, `missing`
is a boolean mask where true means the data_ is missing and `mask` is a boolean mask
where true means the data_ location is valid, e.g. sea points for sea surface temperature.

The mask variable can also be ommited. In this case a grid point is considered valid, if
the grid point has at least 5% of non-clouded data_ (parameter `minfrac`).
从 NetCDF 文件 'fname' 加载变量 'varname'。变量“lon”是
以东度为单位的经度，“纬度”是北纬，“时间”是
一个 Numpy 日期时间向量，“data_full”是一个包含数据的 3-D 数组，“缺失”
是一个布尔掩码，其中 true 表示数据丢失，“掩码”是一个布尔掩码
其中 true 表示数据位置有效，例如海面温度的海点。

掩码变量也可以省略。如果网格点至少具有 5% 的非云数据（参数“minfrac”）在这种情况下，网格点被视为有效，。

At the bare-minimum a NetCDF file should have the following variables and
attributes:
NetCDF 文件至少应具有以下变量和属性：

    netcdf file.nc {
    dimensions:
            time = UNLIMITED ; // (5266 currently)
            lat = 112 ;
            lon = 112 ;
    variables:
            double lon(lon) ;
            double lat(lat) ;
            double time(time) ;
                    time:units = "days since 1900-01-01 00:00:00" ;
            int mask(lat, lon) ;
            float SST(time, lat, lon) ;
                    SST:_FillValue = -9999.f ;
    }

"""
    ds = Dataset(fname);  #打开一个文件
    # lon = ds.variables["lon"][:].data_;   #读取lon经度变量，师姐的数据集，这个长度是8640
    # lat = ds.variables["lat"][:].data_;   #读取lat纬度变量，师姐的数据集，这个长度是4320
    '''测试改的'''
    # lon = ds.variables["lon"][3000:3100].data_;
    # lat = ds.variables["lat"][3000:3100].data_;

    # lon = ds.variables["lon"][:400].data_;
    # lat = ds.variables["lat"][:400].data_;

    data=ds.variables[varname]
    print("ds.variable",data.shape)

    lon = ds.variables["lon"][:].data;
    lat = ds.variables["lat"][:].data;

    time = num2date(ds.variables["time"][:],ds.variables["time"].units);  # 将时间转化为可辨识的格式

    # data_ = ds.variables[varname][:,:,:];  # 海表面温度，师姐的数据集，shape:(1,4320,8640)
    '''测试改的'''
    # data_ = ds.variables[varname][:, 3000:3100, 3000:3100];
    # data_ = ds.variables[varname][:, :, :400, :400][0];  # sst: shape:(1,1,4320,8640),多了一个1，给它去掉变成(1,4320,8640)

    # print(ds.variables[varname].shape) (1,4320,8640)
    # 现在数据有三维，但是我们取了四维，[:, :, :, :]，所以报错，更改下行代码。
    # 之前为什么取4维，因为数据集的形状是(1,1,4320,8640),现在的形状是(1,4320,8640)
    # data_ = ds.variables[varname][:, :, :, :][0];
    data = ds.variables[varname][:, :, :];
    print(data.shape)

    if "mask" in ds.variables:
        mask = ds.variables["mask"][:,:].data == 1;
        #“掩码”是一个布尔掩码其中 true 表示数据位置有效，例如海面温度的海点。
    else:
        print("compute mask for ",varname,": sea point should have at least ",
              minfrac," for valid data_ tought time")
        # 如果网格点至少具有 5% 的非云数据（参数“minfrac”）在这种情况下，网格点被视为有效。
        if np.isscalar(data.mask):   #逻辑函数，如果输入num的类型为标量，则返回true
            mask = np.ones((data.shape[1],data.shape[2]),dtype=np.bool)
            #np.ones()函数返回给定形状和数据类型的新数组，其中元素的值设置1
        else:
            mask = np.mean(~data.mask,axis=0) > minfrac  #求每个数据掩码的均值 axis=0，输出矩阵是1行，求每一列的平均


        print("mask: sea points ",np.sum(mask))  #输出海点的掩码
        print("mask: land points ",np.sum(~mask)) #输出陆地点的掩码

    print("varname ",varname,mask.shape)
    ds.close()

    if np.isscalar(data.mask):     #np.isscalar()是一个逻辑函数，如果输入num的类型为标量，则返回true
        missing = np.zeros(data.shape,dtype=np.bool)   #返回来一个给定形状和类型的用0填充的数组
    else:
        missing = data.mask

    print("data_ shape: ",data.shape)    #输出数据形状
    print("data_ range: ",data.min(),data.max())    #输出数据最小值和最大值

    return lon,lat,time,data,missing,mask


def data_generator(lon,lat,time,data_full,missing,mask,
                   train = True,
                   ntime_win = 3,
                   obs_err_std = 1.,     #用来统计气象观测误差
                   jitter_std = 0.05):

    return data_generator_list(lon,lat,time,[data_full],[missing],mask,
                   train = True,
                   ntime_win = ntime_win,
                   obs_err_std = [obs_err_std],
                   jitter_std = [jitter_std])

def data_generator_list(lon,lat,time,data_full,missing,mask,
                   train = True,
                   ntime_win = 3,
                   obs_err_std = [1.],
                   jitter_std = [0.05]):
    # print("data_full.shape",data_full.shape)
    # print("missing",missing)
    """
Return a generator for training (`train = True`) or testing (`train = False`)
the neural network. `obs_err_std` is the error standard deviation of the
observations. The variable `lon` is the longitude in degrees east, `lat` is the
latitude in degrees North, `time` is a numpy datetime vector, `data_full` is a
3-d array with the data_ and `missing` is a boolean mask where true means the data_ is
missing. `jitter_std` is the standard deviation of the noise to be added to the
data_ during training.
返回用于训练（“train = True”）或测试（“train = False”）的生成器神经网络。“obs_err_std”是错误标准差观察。
变量“lon”是东经度，“lat”是纬度以北度为单位，“time”是一个数字日期时间向量，“data_full”是一个
包含数据的 3-D 数组，“缺失”是一个布尔掩码，其中 true 表示数据是失踪。“jitter_std”是要添加到训练期间的数据。

The output of this function is `datagen`, `ntime` and `meandata`. `datagen` is a
generator function returning a single image (relative to the mean `meandata`),
`ntime` the number of time instances for training or testing and `meandata` is
the temporal mean of the data_.
此函数的输出是“datagen”、“ntime”和“meandata”。“数据生成”是一个返回单个图像的生成器函数（相对于平均值“平均值”），
“ntime”是用于训练或测试的时间实例数，“平均数据”是数据的时间平均值。

    # number of time instances, must be odd
    ntime_win = 3

"""
    # data_full=data_full[0]
    # missing=missing[0]
    # sz = data_full.shape #(96, 64, 64)
    # print("data_full",data_full[0].type())
    sz = data_full[0].shape #(96, 64, 64)
    print("show:",data_full[0])
    print("sz ",sz)
    ntime = sz[0] # 96
    # data_full = data_full[0]
    # missing=missing[0]
    ndata = len(data_full) # 96
    print("ndata",ndata)
    dayofyear = time  #一年中的一天
    dayofyear_cos = np.cos(2 * np.pi * dayofyear/365.25)   #余弦值
    dayofyear_sin = np.sin(2 * np.pi * dayofyear/365.25)   #正弦值


    meandata = [None] * ndata
    data = [None] * ndata
    obs_err_std = obs_err_std *ndata
    #
    # print("ndata",ndata)
    for i in range(ndata):
        meandata[i] = data_full[i].mean(axis=0,keepdims=True)
        data[i] = data_full[i] - meandata[i]   #减去平均值，去噪

        if data_full[i].shape != data_full[0].shape:
            raise ArgumentError("shape are not coherent")



    # scaled mean and inverse of error variance for every input data_
    #每个输入数据的标度平均值和误差方差反比
    # plus lon, lat, cos(time) and sin(time)
    x = np.zeros((sz[0],sz[1],sz[2],2*ndata + 4),dtype="float32")
    print("missing::::::", np.array(missing).shape)
    print("missing::::::", missing[0])
    for i in range(ndata):
        print("data_[i]",np.sum(data[i].filled(0)==0))
        # print("mask", mask)
        x[:,:,:,2*i] = data[i].filled(0) / (obs_err_std[i]**2)
        # x[:,:,:,2*i+1] = (1-data_[i].mask) / (obs_err_std[i]**2)  # error variance
        x[:,:,:,2*i+1] = (1-missing[0]) / (obs_err_std[i]**2)  # error variance




    # scale between -1 and 1
    lon_scaled = 2 * (lon - np.min(lon)) / (np.max(lon) - np.min(lon)) - 1
    lat_scaled = 2 * (lat - np.min(lat)) / (np.max(lat) - np.min(lat)) - 1

    i = 2*ndata
    # x[:,:,:,i  ] = lon_scaled.reshape(1,1,len(lon))
    x[:,:,:,i  ] = lon_scaled.reshape(lon.shape[0],1,lon.shape[1])
    # x[:,:,:,i+1] = lat_scaled.reshape(1,len(lat),1)
    # x[:,:,:,i+1] = lat_scaled.reshape(1,len(lat[0]),1)
    x[:,:,:,i+1] = lat_scaled.reshape(lon.shape[0],1,lon.shape[1])
    x[:,:,:,i+2] = dayofyear_cos.reshape(len(dayofyear_cos),1,1)
    x[:,:,:,i+3] = dayofyear_sin.reshape(len(dayofyear_sin),1,1)

    nvar = 2 * ntime_win * ndata + 4

    missing = np.array(missing, np.bool)  ##后加的
    # generator for data_
    def datagen():
        for i in range(ntime):
            xin = np.zeros((sz[1],sz[2],nvar),dtype="float32")
            xin[:,:,0:(2*ndata + 4)]  = x[i,:,:,:]

            ioffset = (2*ndata + 4)
            for time_index in range(0,ntime_win):
                # nn is centered on the current time, e.g. -1 (past), 0 (present), 1 (future)
                nn = time_index - (ntime_win//2)
                # current time is already included, skip it
                if nn != 0:
                    i_clamped = min(ntime-1,max(0,i+nn))
                    xin[:,:,ioffset:(ioffset + 2*ndata)] = x[i_clamped,:,:,0:(2*ndata)]
                    ioffset = ioffset + 2*ndata

            # add missing data_ during training randomly
            # print("ndata:",ndata)
            # print("missing.shape", len(missing))
            if train:
                #imask = random.randrange(0,missing.shape[0])
                imask = random.randrange(0,ntime)

                # print('missing.shape', len(missing))


                for j in range(ndata):
                    selmask = missing[j][imask,:,:]
                    # print('selmask', np.sum(selmask==True))

                    xin[:,:,2*j][selmask] = 0
                    xin[:,:,2*j+1][selmask] = 0

                # add jitter
                for j in range(ndata):
                    xin[:,:,2*j] += jitter_std[j] * np.random.randn(sz[1],sz[2])
                    xin[:,:,2*j + 2*ndata + 4] += jitter_std[j] * np.random.randn(sz[1],sz[2])
                    xin[:,:,2*j + 4*ndata + 4] += jitter_std[j] * np.random.randn(sz[1],sz[2])

            yield (xin,x[i,:,:,0:2])

    # meandata[0] is the primary variable to be reconstructed
    return datagen,nvar,ntime,meandata[0]



def savesample(fname,m_rec,σ2_rec,meandata,lon,lat,e,ii,offset,
               transfun = (identity, identity)):
    fill_value = -9999.
    recdata = transfun[1](m_rec  + meandata)
    # todo apply transfun to sigma_rec

    if transfun[1] == np.exp:
        # relative error
        #sigma_rec = recdata * np.sqrt(σ2_rec)
        sigma_rec = np.sqrt(σ2_rec) # debug
    elif transfun[1] == identity:
        sigma_rec = np.sqrt(σ2_rec)
    else:
        print("warning: sigma_rec is not transformed")
        sigma_rec = np.sqrt(σ2_rec)


    if ii == 0:
        # create file
        root_grp = Dataset(fname, 'w', format='NETCDF4')

        # dimensions
        root_grp.createDimension('time', None)
        root_grp.createDimension('lon', len(lon))
        root_grp.createDimension('lat', len(lat))

        # variables
        nc_lon = root_grp.createVariable('lon', 'f4', ('lon',))
        nc_lat = root_grp.createVariable('lat', 'f4', ('lat',))
        nc_meandata = root_grp.createVariable(
            'meandata', 'f4', ('lat','lon'),
            fill_value=fill_value)

        nc_mean_rec = root_grp.createVariable(
            'mean_rec', 'f4', ('time', 'lat', 'lon'),
            fill_value=fill_value)

        nc_sigma_rec = root_grp.createVariable(
            'sigma_rec', 'f4', ('time', 'lat', 'lon',),
            fill_value=fill_value)

        # data_
        # nc_lon[:] = lon
        # nc_lat[:] = lat
        # nc_meandata[:,:] = meandata
    else:
        # append to file
        root_grp = Dataset(fname, 'a')
        nc_mean_rec = root_grp.variables['mean_rec']
        nc_sigma_rec = root_grp.variables['sigma_rec']

    for n in range(m_rec.shape[0]):
    # nc_mean_rec[n+offset,:,:] = np.ma.masked_array(
    # recdata[n,:,:],meandata.mask)
        nc_mean_rec[n+offset,:,:] = np.ma.masked_array(
            recdata[n,:,:],meandata.mask)
        nc_sigma_rec[n+offset,:,:] = np.ma.masked_array(
            sigma_rec[n,:,:],meandata.mask)


    root_grp.close()


# save inversion
def sinv(x, minx = 1e-3):
    return 1 / tf.maximum(x,minx)
    _, batch_RMS,batch_R2 = reconstruct(
        train_lon_all, train_lat_all, train_mask_all, meandata,
        train_datagen, train_len,
        test_datagen, test_len,
        outdir,
        # data_epoch=i,
        # batch_size=1,
        # shuffle_buffer_size=1,
        # enc_nfilter_internal=[8, 12, 18, 27],
        transfun=transfun,
        nvar=nvar,
        **kwargs)
def reconstruct(lon,lat,mask,meandata,test_meandata, #东经度，以北度为单位的维度，布尔掩码其中true表示数据位置有效，数据的时间平均值
                train_datagen,train_len, #生成器函数返回单个图像进行训练，训练图像的数量
                test_datagen,test_len, #生成器函数返回单个图像进行测试，测试图像的数量
                outdir, N_S_ratio,corrup_rate, #输出目录
                resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR, #[TensorFlow]中定义的调整大小方法之一
                data_epoch = 0,
                epochs = 200,  #用于训练神经网络的epoch数
                batch_size = 1, #小批量的大小
                save_each = 10,  #每个纪元重建缺失的数据。如果"save_each"为0，则禁用重复保存。最后一个纪元总是被保存下来
                save_model_each = 50, #每隔一段时间保存神经网络的一个检查点
                skipconnections = [1,2,3,4], #卷积层的索引列表
                dropout_rate_train = 0.3,  #训练期间辍学的概率
                tensorboard = True, #激活张量板诊断
                truth_uncertain = False, #你对感知到的真相有多确定
                shuffle_buffer_size = 3*15,  #随机缓冲区的图像数量
                nvar = 10,  #输入变量的数量
                enc_nfilter_internal = [16,24,36,54], #内部卷积层的过滤器数量（在输入卷积层之后）
                frac_dense_layer = [0.2],
                clip_grad = 5.0,  #将裁剪梯度剪切到最大L2标准
                regularization_L2_beta = 0,  #标量以强制权重L2正则化
                transfun = (identity,identity),
                savesample = savesample,
                learning_rate = 0.00003,  #初始学习率
                learning_rate_decay_epoch = np.inf,  #倾斜率的指数回报率。之后，学习率减半
                iseed = None,
                nprefetch = 0,
                loss = [],
                nepoch_keep_missing = 0,
):
    """
Train a neural network to reconstruct missing data_ using the training data_ set
and periodically run the neural network on the test dataset. The function returns the
filename of the latest reconstruction.
训练神经网络以使用训练数据集重建缺失数据
并定期在测试数据集上运行神经网络。该函数返回最新重建的文件名。
## Parameters

 * `lon`: longitude in degrees East  东经度
 * `lat`: latitude in degrees North
 * `mask`:  boolean mask where true means the data_ location is valid,
e.g. sea points for sea surface temperature.
 * `meandata`: the temporal mean of the data_.
 * `train_datagen`: generator function returning a single image for training
 * `train_len`: number of training images
 * `test_datagen`: generator function returning a single image for testing
 * `test_len`: number of testing images
 * `outdir`: output directory

## Optional input arguments

 * `resize_method`: one of the resize methods defined in [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/image/resize_images)
 * `epochs`: number of epochs for training the neural network
 * `batch_size`: size of a mini-batch
 * `save_each`: reconstruct the missing data_ every `save_each` epoch. Repeated saving is disabled if `save_each` is zero. The last epoch is always saved.
 * `save_model_each`: save a checkpoint of the neural network every
      `save_model_each` epoch
 * `skipconnections`: list of indices of convolutional layers with
     skip-connections
 * `dropout_rate_train`: probability for drop-out during training
 * `tensorboard`: activate tensorboard diagnostics
 * `truth_uncertain`: how certain you are about the perceived truth?
 * `shuffle_buffer_size`: number of images for the shuffle buffer
 * `nvar`: number of input variables
 * `enc_nfilter_internal`: number of filters for the internal convolutional layers
      (after the input convolutional layer)
 * `clip_grad`: clip gradient to a maximum L2-norm.
 * `regularization_L2_beta`: scalar to enforce L2 regularization on the weight
 * `learning_rate`:  The initial learning rate
 * `learning_rate_decay_epoch`: The exponential recay rate of the leaning rate. After `learning_rate_decay_epoch` the learning rate is halved. The learning rate is compute as  `learning_rate * 0.5^(epoch / learning_rate_decay_epoch)`. `learning_rate_decay_epoch` can be `numpy.inf` for a constant learning rate (default)
"""

    mim_loss = 1
    mim_rmse = 100
    mim_mse = 100
    mim_mae = 10
    mim_r2 = 0
    mim_ssim = 0
    mim_psnr = 0

    mim_rmse_epoch = 0
    mim_mse_epoch = 0
    mim_mae_epoch = 0
    mim_r2_epoch = 0
    mim_ssim_epoch = 0
    mim_psnr_epoch = 0

    mim_rmse_batch = 0
    mim_mse_batch = 0
    mim_mae_batch = 0
    mim_r2_batch = 0
    mim_ssim_batch = 0
    mim_psnr_batch = 0
    min_mean_mse = 10
    min_mean_rmse = 100
    min_mean_mae = 100
    max_mean_r2 = 0
    max_mean_ssim = 0
    max_mean_psnr = 0
    min_mean_mse_epoch = 0
    min_mean_rmse_epoch = 0
    min_mean_mae_epoch = 0
    max_mean_r2_epoch = 0
    max_mean_ssim_epoch = 0
    max_mean_psnr_epoch = 0
    rusult = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if iseed != None:
        np.random.seed(iseed)
        tf.compat.v1.set_random_seed(np.random.randint(0,2**32-1))
        random.seed(np.random.randint(0,2**32-1))

    print("regularization_L2_beta ",regularization_L2_beta)
    print("enc_nfilter_internal ",enc_nfilter_internal)
    print("nvar ",nvar)
    print("nepoch_keep_missing ",nepoch_keep_missing)

    # number of output variables
    nvarout = 2
    enc_nfilter = [nvar] + enc_nfilter_internal
    dec_nfilter = enc_nfilter_internal[::-1] + [nvarout]
    # old behaviour
    #dec_nfilter = enc_nfilter_internal[::-1] + [nvar]

    if outdir != None:
        if not os.path.exists(os.path.join(outdir, str(data_epoch))):
            # os.path.join(outdir, str(data_epoch)
            os.mkdir(os.path.join(outdir, str(data_epoch)))

    # jmax,imax = mask.shape
    batch, jmax,imax = mask.shape

    sess = tf.compat.v1.Session()
   

    # Repeat the input indefinitely.
    # training dataset iterator
    train_dataset = tf.data.Dataset.from_generator(
        train_datagen, (tf.float32,tf.float32),
        (tf.TensorShape([jmax,imax,nvar]),tf.TensorShape([jmax,imax,2]))).repeat().batch(batch_size)

    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    # train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    # train_iterator = iter(train_dataset)
    print("dir", dir(train_iterator))
    train_iterator_handle = sess.run(train_iterator.string_handle())
    # train_iterator_handle =next(train_iterator)  # 创建一个迭代器
    # train_iterator_handle = train_iterator.get_next()
    # test dataset without added clouds
    # must be reinitializable
    test_dataset = tf.data.Dataset.from_generator(
        test_datagen, (tf.float32,tf.float32),
        (tf.TensorShape([jmax,imax,nvar]),tf.TensorShape([jmax,imax,2]))).batch(batch_size)

    if nprefetch > 0:
        train_dataset = train_dataset.prefetch(nprefetch)
        test_dataset = test_dataset.prefetch(nprefetch)

    test_iterator = tf.compat.v1.data.Iterator.from_structure(test_dataset.output_types,test_dataset.output_shapes)
    # test_iterator = iter(test_dataset)
    # test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)

    test_iterator_init_op = test_iterator.make_initializer(test_dataset)
    # test_iterator_init_op = None

    test_iterator_handle = sess.run(test_iterator.string_handle())
    # test_iterator_handle =next(test_iterator)  # 创建一个迭代器
    # test_iterator_handle = test_iterator.get_next()
    handle = tf.compat.v1.placeholder(tf.string, shape=[], name = "handle_name_iterator")
    iterator = tf.compat.v1.data.Iterator.from_string_handle(
            handle, train_iterator.output_types, output_shapes = train_iterator.output_shapes)


    inputs_,xtrue = iterator.get_next() #  yield (xin,x[i,:,:,0:2])
    print("input shape:", inputs_.shape)
    print("xtrue shape:", xtrue.shape)

    # activation function for convolutional layer
    conv_activation=tf.nn.leaky_relu
    #conv_activation=tf.nn.relu

    # Encoder

    enc_nlayers = len(enc_nfilter)
    enc_conv = [None] * enc_nlayers
    enc_avgpool = [None] * enc_nlayers

    enc_avgpool[0] = inputs_

    for l in range(1,enc_nlayers):
        # enc_conv[l] = tf.compat.v1.layers.conv2d(enc_avgpool[l-1],
        #                                enc_nfilter[l],
        #                                (3,3),
        #                                padding='same',
        #                                activation=conv_activation)
        enc_conv[l] = tf.keras.layers.Conv2D(enc_nfilter[l],
                                             (3, 3),
                                             padding='same',
                                             activation=conv_activation)(enc_avgpool[l - 1])
        print("encoder: output size of convolutional layer: ",l,enc_conv[l].shape)

        # enc_avgpool[l] = tf.compat.v1.layers.average_pooling2d(enc_conv[l],
        #                                              (2,2),
        #                                              (2,2),
        #                                              padding='same')
        enc_avgpool[l] = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(enc_conv[l])

        print("encoder: output size of pooling layer: ",l,enc_avgpool[l].shape)

        enc_last = enc_avgpool[-1]

    # default is no drop-out
    dropout_rate = tf.compat.v1.placeholder_with_default(0.0, shape=())

    if len(frac_dense_layer) == 0:
        dense_2d = enc_last
    else:
        # Dense Layers
        ndensein = enc_last.shape[1:].num_elements()
        print("ndensein ",ndensein)

        avgpool_flat = tf.reshape(enc_last, [-1, ndensein])

        # number of output units for the dense layers
        dense_units = [floor(ndensein*frac) for frac in frac_dense_layer + list(reversed(frac_dense_layer[:-1]))]
        # last dense layer must give again the same number as input units
        dense_units.append(ndensein)

        dense = [None] * (4*len(frac_dense_layer)+1)
        dense[0] = avgpool_flat

        for i in range(2*len(frac_dense_layer)):
            dense[2*i+1] = tf.compat.v1.layers.dense(inputs=dense[2*i],
                                           units=dense_units[i],
                                           activation=tf.nn.relu)

            print("dense layer: output units: ",i,dense[2*i+1].shape)
            dense[2*i+2] = tf.compat.v1.layers.dropout(inputs=dense[2*i+1], rate=dropout_rate)

        dense_2d = tf.reshape(dense[-1], tf.shape(input=enc_last))

    ### Decoder
    dec_conv = [None] * enc_nlayers
    dec_upsample = [None] * enc_nlayers

    dec_conv[0] = dense_2d

    for l in range(1,enc_nlayers):
        l2 = enc_nlayers-l

        dec_upsample[l] = tf.image.resize(
            dec_conv[l-1],
            enc_conv[l2].shape[1:3],
            method=resize_method)
        print("decoder: output size of upsample layer: ",l,dec_upsample[l].shape)

        # short-cut
        if l in skipconnections:
            print("skip connection at ",l)
            dec_upsample[l] = tf.concat([dec_upsample[l],enc_avgpool[l2-1]],3)
            print("decoder: output size of concatenation: ",l,dec_upsample[l].shape)

        dec_conv[l] = tf.compat.v1.layers.conv2d(
            dec_upsample[l],
            dec_nfilter[l],
            (3,3),
            padding='same',
            activation=conv_activation)

        print("decoder: output size of convolutional layer: ",l,dec_conv[l].shape)

    # last layer of decoder
    xrec = dec_conv[-1]#第26层的输出，(?, 64, 64, 2)

    """
    xrec[:,:,:,0] 表示为 温度异常 除以误差方差
    xrec[:,:,:,1] 表示为 预期误差的方差的倒数的对数

    xtrue[:,:,:,0] 表示为 温度异常 除以 误差方差
    xtrue[:,:,:,1] 表示为 误差方差的倒数
    """
    loginvσ2_rec = xrec[:,:,:,1]
    invσ2_rec = tf.exp(tf.minimum(loginvσ2_rec,10))


    σ2_rec = sinv(invσ2_rec) #预测的误差方差
    m_rec = xrec[:,:,:,0] * σ2_rec #预测的SST异常

    σ2_true = sinv(xtrue[:,:,:,1]) #真实误差方差
    m_true = xtrue[:,:,:,0] * σ2_true # 真实的SST异常


    σ2_in = sinv(inputs_[:,:,:,1])
    m_in = inputs_[:,:,:,0] * σ2_in


    # print("σ2_true.shape", σ2_true.shape)
    # print("xtrue.shape", xtrue.shape)
    # print("m_rec",m_rec.shape)
    ######ZZJ 写
    # print('m_rec.shape',m_rec.shape)
    # print(' xtrue[:,:,:,0].shape:',  xtrue[:,:,:,0].shape)
    # print('meandata.shape:', meandata.shape)
    obs_err_std = [1.]

    recons_data = xrec[:,:,:,0] + meandata
    ground_data = xtrue[:,:,:,0] + meandata

    ####### ZZJ 写


    difference = recons_data - ground_data
    print("recons_data.shape",recons_data.shape)
    print("ground_data.shape",ground_data.shape)
    mask_issea = tf.compat.v1.placeholder(
        tf.float32,
        # shape = (mask.shape[0], mask.shape[1]),
        shape = (mask.shape[0], mask.shape[1],mask.shape[2]),
        name = "mask_issea")

    # 1 if measurement
    # 0 if no measurement (cloud or land for SST)
    mask_noncloud = tf.cast(tf.math.logical_not(tf.equal(xtrue[:,:,:,1], 0)),xtrue.dtype)

    """
    tf.equal(xtrue[:,:,:,1], 0)) = 误差方差中如果数值缺失，则为0，tf.equal判断如果是0，则返回true，也就是说缺的地方是true
    tf.math.logical_not(tf.equal(xtrue[:,:,:,1], 0)) = 将结果反过来， true变false， false变true，也就是说不缺的地方是true
    tf.cast(tf.math.logical_not(tf.equal(xtrue[:,:,:,1], 0)), xtrue.dtype)  表示 
    """
    n_noncloud = tf.reduce_sum(input_tensor=mask_noncloud)
    print("n_noncloud",n_noncloud.shape)
    print("σ2_true",σ2_true.shape)
    if truth_uncertain:
        # KL divergence between two univariate Gaussians p and q
        # p ~ N(σ2_1,\mu_1)
        # q ~ N(σ2_2,\mu_2)
        #
        # 2 KL(p,q) = log(σ2_2/σ2_1) + (σ2_1 + (\mu_1 - \mu_2)^2)/(σ2_2) - 1
        # 2 KL(p,q) = log(σ2_2) - log(σ2_1) + (σ2_1 + (\mu_1 - \mu_2)^2)/(σ2_2) - 1
        # 2 KL(p_true,q_rec) = log(σ2_rec/σ2_true) + (σ2_true + (\mu_rec - \mu_true)^2)/(σ2_rec) - 1

        cost = (tf.reduce_sum(input_tensor=tf.multiply(
            tf.math.log(σ2_rec/σ2_true) + (σ2_true + difference**2) / σ2_rec,mask_noncloud))) / n_noncloud
    else:
        cost = (tf.reduce_sum(input_tensor=tf.multiply(tf.math.log(σ2_rec),mask_noncloud)) +
            tf.reduce_sum(input_tensor=tf.multiply(difference**2 / σ2_rec,mask_noncloud))) / n_noncloud


    # L2 regularization of weights
    if regularization_L2_beta != 0:
        trainable_variables   = tf.compat.v1.trainable_variables()
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in trainable_variables
                            if 'bias' not in v.name ]) * regularization_L2_beta
        cost = cost + lossL2

    mask_cloud = tf.cast(tf.equal(xtrue[:,:,:,1], 0), xtrue.dtype)
    n_cloud = tf.reduce_sum(input_tensor=mask_cloud)
    RMS = tf.sqrt(tf.reduce_sum(input_tensor=tf.multiply(difference**2,mask_cloud))
                  / n_cloud)
    MAE_socre = tf.reduce_sum(input_tensor=tf.multiply(tf.abs(difference), mask_cloud)) / n_cloud
    MSE_socre = tf.reduce_sum(input_tensor=tf.multiply(difference ** 2, mask_cloud)) / n_cloud

    SSE = tf.reduce_sum((difference ** 2) * mask_cloud)
    SST = tf.reduce_sum((recons_data - (tf.reduce_sum(ground_data * mask_cloud) / (tf.reduce_sum(mask_cloud))) * mask_cloud) ** 2)
    R2_socre = 1 - SSE / SST

    # mse_score = torch.sum((difference * mask_cloud) ** 2) / torch.sum(mask_cloud)
    img1 = tf.expand_dims(recons_data, axis=-1)  # 扩展为 (batch_size, height, width, 1)
    img2 = tf.expand_dims(ground_data, axis=-1)  # 扩展为 (batch_size, height, width, 1)
    SSIM_socre= tf.image.ssim(img1, img2, max_val=255.0)
    # SSIM_socre= pytorch_ssim.ssim(recons_data, ground_data)
    #
    # PSNR_socre= R2_socre
    # PSNR_socre= 20 * torch.log10(255 / torch.sqrt(MSE_socre))
    PSNR_socre = 20 * tf.math.log(255.0 / tf.sqrt(MSE_socre)) / tf.math.log(10.0)
    mask_type = "Cloud_mask"  # 可选 Cloud_mask/Square_mask/Strip_mask (函数内部可以选mask的位置，默认为right)


    # to debug
    # cost = RMS
    print(f'm_rec.shape:{m_rec.shape}')
    if tensorboard:
        with tf.compat.v1.name_scope('Validation'):
            tf.compat.v1.summary.scalar('RMS', RMS)
            tf.compat.v1.summary.scalar('cost', cost)

            # tf.compat.v1.summary.image("m_rec",tf.expand_dims(
            #     tf.reverse(tf.multiply(m_rec,mask_issea),[1]),-1))
            tf.compat.v1.summary.image("m_rec", tf.expand_dims(
                tf.reverse(m_rec, [1]), -1))

            # tf.compat.v1.summary.image("m_true",tf.expand_dims(
            #     tf.reverse(tf.multiply(m_true,mask_issea),[1]),-1))
            tf.compat.v1.summary.image("m_true", tf.expand_dims(
                tf.reverse(m_true, [1]), -1))

            # tf.compat.v1.summary.image("sigma2_rec",tf.expand_dims(
            #     tf.reverse(tf.multiply(σ2_rec,mask_issea),[1]),-1))

    # parameters for Adam optimizer (default values)
    #learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08

    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = 1e-3
    # learning_rate = tf.train.exponential_decay(starter_learning_rate,
    #                                                      global_step,
    #                                                      50, 0.96, staircase=True)

    learning_rate_ = tf.compat.v1.placeholder(tf.float32, shape=[])

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_,beta1,beta2,epsilon)
    gradients, variables = zip(*optimizer.compute_gradients(cost))
    gradients, _ = tf.clip_by_global_norm(gradients, clip_grad)
    opt = optimizer.apply_gradients(zip(gradients, variables))

    # Passing global_step to minimize() will increment it at each step.
    # opt = (
    #     tf.train.GradientDescentOptimizer(learning_rate)
    #     .minimize(cost, global_step=global_step)
    # )

    # optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)


    dt_start = datetime.now()
    print(dt_start)

    if tensorboard:
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(outdir + '/train',
                                          sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(outdir + '/test')
    else:
        # unused
        merged = tf.constant(0.0, shape=[1], dtype="float32")

    index = 0

    print("init")
    sess.run(tf.compat.v1.global_variables_initializer())
    logger.debug('init done')

    saver = tf.compat.v1.train.Saver()

    # final output file name
    fname = None

    # loop over epochs
    for e in range(epochs):
        if nepoch_keep_missing > 0:
            # use same clouds for every e.g. 20 epochs
            random.seed(iseed + e//nepoch_keep_missing)


        # loop over training datasets
        for ii in range(ceil(train_len / batch_size)):
            # run a single step of the optimizer
            #logger.debug(f'running {ii}')
            summary, batch_cost, batch_RMS, bs, batch_learning_rate,batch_m_rec, batch_m_true,_ = sess.run(
                [merged, cost, RMS, mask_noncloud, learning_rate_,recons_data,ground_data,opt],feed_dict={
                    handle: train_iterator_handle,
                    mask_issea: mask,
                    learning_rate_: learning_rate * (0.5 ** (e / learning_rate_decay_epoch)),
                    dropout_rate: dropout_rate_train})

            Visual_epoch = save_each
            if e > (Visual_epoch - 1) and e < (Visual_epoch * 2) and e % Visual_epoch == 0:
                visua_and_save('Train', e, ii, batch_m_rec, 'recons', N_S_ratio, mask_type, corrup_rate)
                visua_and_save('Train', e, ii, batch_m_true, 'ground_recons', N_S_ratio, mask_type, corrup_rate)
            elif e > (Visual_epoch * 2 - 1) and e % (Visual_epoch) == 0:
                visua_and_save('Train', e, ii, batch_m_rec, 'recons', N_S_ratio, mask_type, corrup_rate)

            #logger.debug('running done')
            loss.append(batch_cost)

            if tensorboard:
                train_writer.add_summary(summary, index)

            index += 1

            if ii % 20 == 0:
            #if ii % 1 == 0:
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Training loss: {:.20f}".format(batch_cost),
                      "RMS: {:.20f}".format(batch_RMS), batch_learning_rate )


        if ((e == epochs-1) or ((save_each > 0) and (e % save_each == 0))) and outdir != None:
            print("Save output",e)

            timestr = datetime.now().strftime("%Y-%m-%dT%H%M%S")
            fname = os.path.join(outdir, str(data_epoch), "data_-{}.nc".format(timestr))

            # reset test iterator, so that we start from the beginning
            sess.run(test_iterator_init_op)
            total_mse = []
            total_rmse = []
            total_mae = []
            total_r2 = []
            total_ssim = []
            total_psnr = []
            for ii in range(ceil(test_len / batch_size)):
                summary, batch_cost,batch_RMS,batch_R2,batch_MAE,batch_MSE,batch_SSIM,batch_PSNR,batch_m_rec,batch_σ2_rec,batch_m_true= sess.run(
                    [merged, cost,RMS,R2_socre, MAE_socre, MSE_socre, SSIM_socre, PSNR_socre, recons_data,σ2_rec,ground_data],
                    feed_dict = { handle: test_iterator_handle,
                                  mask_issea: mask })

                # print("Test:RMS: {:.20f}|Test:R2: {:.20f}".format(batch_RMS,batch_R2))
                # time instances already written
                offset = ii*batch_size
                # savesample(fname,batch_m_rec,batch_σ2_rec,meandata,lon,lat,e,ii,
                #            offset, transfun = transfun)
                total_rmse.append(batch_RMS)
                total_mse.append(batch_MSE)
                total_mae.append(batch_MAE)
                total_r2.append(batch_R2)
                total_ssim.append(batch_SSIM)
                total_psnr.append(batch_PSNR)
                # print("batch_m_rec:",batch_m_rec.type())
                # print("batch_m_rec:",batch_m_rec.shape)
                # visua_and_save('Train', 1, i, m_true, 'real_week', N_S_ratio, mask_type, corrup_rate)
                Visual_epoch = save_each
                if e > (Visual_epoch - 1) and e < (Visual_epoch * 2) and e % Visual_epoch == 0:
                    visua_and_save('Test', e, ii, batch_m_rec, 'valid_recons', N_S_ratio, mask_type, corrup_rate)
                    visua_and_save('Test', e, ii, batch_m_true, 'valid_ground_recons', N_S_ratio, mask_type, corrup_rate)
                    # visua_and_save('Test', e, ii, true, 'ground_recons', N_S_ratio, mask_type, corrup_rate)
                elif e > (Visual_epoch * 2 - 1) and e % (Visual_epoch) == 0:
                    visua_and_save('Test', e, ii, batch_m_rec, 'valid_recons', N_S_ratio, mask_type, corrup_rate)
                Visual_epoch_vaild = save_each*3
                if e > (Visual_epoch_vaild - 1) and (mim_rmse >= batch_RMS):  # valid_mse
                    mim_rmse = batch_RMS
                    mim_rmse_epoch = e
                    mim_rmse_batch = i
                if e > (Visual_epoch_vaild - 1) and (mim_mse >= batch_MSE):  # valid_mse
                    mim_mse = batch_MSE
                    mim_mse_epoch = e
                    mim_mse_batch = i
                if e > (Visual_epoch_vaild - 1) and (mim_mae >= batch_MAE):  # valid_mse
                    mim_mae = batch_MAE
                    mim_mae_epoch = e
                    mim_mae_batch = i
                if e > (Visual_epoch_vaild - 1) and (mim_r2 <= batch_R2):  # valid_r2
                    mim_r2 = batch_R2
                    mim_r2_epoch = e
                    mim_r2_batch = i
                if e > (Visual_epoch_vaild - 1) and (mim_ssim <= batch_SSIM.item()):  # valid_r2
                    mim_ssim = batch_SSIM.item()
                    mim_ssim_epoch = e
                    mim_ssim_batch = i

                if e > (Visual_epoch_vaild - 1) and (mim_psnr <= batch_PSNR.item()):  # valid_r2
                    mim_psnr = batch_PSNR.item()
                    mim_psnr_epoch = e
                    mim_psnr_batch = i

        if e > (Visual_epoch_vaild - 1) and (min_mean_rmse >= np.mean(total_rmse)):  # valid_mse
            min_mean_rmse = np.mean(total_rmse)
            min_mean_rmse_epoch = e
        if e > (Visual_epoch_vaild - 1) and (min_mean_mse >= np.mean(total_mse)):  # valid_mse
            min_mean_mse = np.mean(total_mse)
            min_mean_mse_epoch = e
        if e > (Visual_epoch_vaild - 1) and (min_mean_mae >= np.mean(total_mae)):  # valid_mse
            min_mean_mae = np.mean(total_mae)
            min_mean_mae_epoch = e
        if e > (Visual_epoch_vaild - 1) and (max_mean_r2 <= np.mean(total_r2)):  # valid_r2
            max_mean_r2 = np.mean(total_r2)
            max_mean_r2_epoch = e
        if e > (Visual_epoch_vaild - 1) and (max_mean_ssim <= np.mean(total_ssim)):  # valid_r2
            max_mean_ssim = np.mean(total_ssim)
            max_mean_ssim_epoch = e
        if e > (Visual_epoch_vaild - 1) and (max_mean_psnr <= np.mean(total_psnr)):  # valid_r2
            max_mean_psnr = np.mean(total_psnr)
            max_mean_psnr_epoch = e

        rusult[0] = min_mean_rmse
        rusult[1] = min_mean_rmse_epoch
        rusult[2] = min_mean_mse
        rusult[3] = min_mean_mse_epoch
        rusult[4] = min_mean_mae
        rusult[5] = min_mean_mae_epoch
        rusult[6] = max_mean_r2
        rusult[7] = max_mean_r2_epoch
        rusult[8] = max_mean_ssim
        rusult[9] = max_mean_ssim_epoch
        rusult[10] = max_mean_psnr
        rusult[11] = max_mean_psnr_epoch
        with open("result_{}_{}".format(N_S_ratio, corrup_rate), 'w') as file:
            file.write("best: Rmse:" + str(rusult[0]) + "[Epoch:" + str(rusult[1]) + "] "
                       + "r2:" + str(rusult[6]) + "[Epoch:" + str(rusult[7]) + "]"
                       + "ssim:" + str(rusult[8]) + "[Epoch:" + str(rusult[9]) + "]"
                       + "psnr:" + str(rusult[10]) + "[Epoch:" + str(rusult[11]) + "]"
                       + "mse:" + str(rusult[2]) + "[Epoch:" + str(rusult[3]) + "] "
                       + "mae:" + str(rusult[4]) + "[Epoch:" + str(rusult[5]) + "] ")
        print("batch最低rmse:{:.8f}  [Epoch{}_batch{}]".format(mim_rmse, mim_rmse_epoch, mim_rmse_batch))
        print("batch最低mse:{:.8f}  [Epoch{}_batch{}]".format(mim_mse, mim_mse_epoch, mim_mse_batch))
        print("batch最低mae:{:.8f}  [Epoch{}_batch{}]".format(mim_mae, mim_mae_epoch, mim_mae_batch))
        print("batch最高r2:{:.8f}  [Epoch{}_batch{}]".format(mim_r2, mim_r2_epoch, mim_r2_batch))
        print("batch最高ssim:{:.8f}  [Epoch{}_batch{}]".format(mim_ssim, mim_ssim_epoch, mim_ssim_batch))
        print("batch最高psnr:{:.8f}  [Epoch{}_batch{}]".format(mim_psnr, mim_psnr_epoch, mim_psnr_batch))
        print('Min_mean_rmse:{:.8f} [Epoch{}]\t Min_mean_mse:{:.8f} [Epoch{}]\t Min_mean_mae:{:.8f} [Epoch{}]\t Max_mean_r2:{:.8f} [Epoch{}]\t Max_mean_ssim:{:.8f} [Epoch{}]\t Max_mean_psnr:{:.8f} [Epoch{}]\t'
            .format(min_mean_rmse, min_mean_rmse_epoch, min_mean_mse, min_mean_mse_epoch, min_mean_mae,
                    min_mean_mae_epoch, max_mean_r2, max_mean_r2_epoch, max_mean_ssim, max_mean_ssim_epoch,
                    max_mean_psnr, max_mean_psnr_epoch))
        if ((save_model_each > 0) and (e % save_model_each == 0)) and outdir != None:
            save_path = saver.save(sess, os.path.join(
                outdir,"model-{:03d}.ckpt".format(e+1)))


    # free all resources associated with the session
    sess.close()

    dt_end = datetime.now()
    print(dt_end)
    print(dt_end - dt_start)

    return fname, batch_RMS,batch_R2
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
def reconstruct_gridded_nc(outdir,train_data_all,
                           train_missing_all, train_mask_all,train_lon_all, train_lat_all, train_time_all,
                           test_data_all,  test_missing_all, test_mask_all,test_lon_all, test_lat_all, test_time_all,mask_path,N_S_ratio,corrup_rate,
                           jitter_std = 0.05,
                           ntime_win = 3,
                           transfun = (identity, identity),
                           **kwargs):
    """
Train a neural network to reconstruct missing data_ from the NetCDF variable
`varname` in the NetCDF file `filename`. Results are saved in the output
directory `outdir`. `jitter_std` is the standard deviation of the noise to be
added to the data_ during training.
See `DINCAE.reconstruct` for other keyword arguments and
`DINCAE.load_gridded_nc` for the NetCDF format.
训练神经网络以从在NETCDF文件名中的 NetCDF变量重建缺失的数据，结果保存在输出目录“outdir”中。
“jitter_std”是在训练期间要添加到数据中的噪声的标准偏差。
有关其他关键字参数，请参阅“DINCAE.reconstruct”;
有关NetCDF格式，请参阅"DINCAE.load_gridded_nc"

"""

    #
    # lat_start = 2953  # 维度起点 -29.875 S  #2941
    # lat_end = 3017  # 维度终点  -32.5416 S
    # lon_start = 2500  # 经度起点  -75.125 W  #2517
    # lon_end = 2564  #经度终点  -72.458 W

    # lat_start = 1280  # 维度起点 26N
    # lat_end = 1344  # 维度终点  23N(22.8)
    # lon_start = 5800  # 经度起点  110E
    # lon_end = 5864  # 经度终点  113E(113.2)
    #
    # data_list, missing_list, mask_list = [], [], [], [], [], []
    #
    # for file in filename:
    #     lon_all, lat_all, time, data_all, missing_all, mask_all = load_gridded_nc(file, varname)
    #
    #     data_list.append(data_all[0,lat_start:lat_end,lon_start:lon_end])
    #     missing_list.append(missing_all[0,lat_start:lat_end,lon_start:lon_end])
    #     mask_list.append(mask_all[lat_start:lat_end,lon_start:lon_end])
    #
    # lon_all = np.stack(lon_list)
    # lat_all = np.stack(lat_list)
    # time_all = np.stack(time_list)

    train_data_all, train_missing_all, train_mask_all, train_lon_all, train_lat_all, train_time_all =train_data_all, train_missing_all, train_mask_all, train_lon_all, train_lat_all, train_time_all
    test_data_all, test_missing_all, test_mask_all, test_lon_all, test_lat_all, test_time_all =test_data_all, test_missing_all, test_mask_all, test_lon_all, test_lat_all, test_time_all
    print("    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("    if np.isscalar(data_.mask):", np.isscalar(train_mask_all))
    # data_all = np.stack(data_list)
    # missing_all = np.stack(missing_list)
    # mask_all = np.stack(mask_list)
    # print("train_data_all.shape",train_data_all.shape)
    # print("lat_all.shape",lat_all.shape)
    # print("time_all.shape",time_all.shape)
    # print("train_data_all.shape",train_data_all.shape)
    # print("test_data_all.shape",test_data_all.shape)
    # print("mask_all.shape",mask_all.shape)
    print("train_data_all.shape", train_data_all.shape)
    print("test_data_all.shape", test_data_all.shape)
    print("train_lon_all.shape", train_lon_all.shape)
    print("train_lat_all.shape", train_lat_all.shape)
    print("test_lon_all.shape", test_lon_all.shape)
    print("test_lat_all.shape", test_lat_all.shape)
    print("train_missing_all.shape", train_missing_all.shape)
    print("test_missing_all.shape", test_missing_all.shape)
    print("train_mask_all.shape", train_mask_all.shape)
    print("test_mask_all.shape", test_mask_all.shape)
    print("train_time_all.shape", train_time_all.shape)
    print("test_time_all.shape", test_time_all.shape)
    """
    train_data_all.shape (96, 64, 64)
    test_data_all.shape (28, 64, 64)
    train_lon_all.shape (96, 64)
    train_lat_all.shape (96, 64)
    test_lon_all.shape (28, 64)
    test_lat_all.shape (28, 64)
    train_missing_all.shape (96, 64, 64)
    test_missing_all.shape (28, 64, 64)
    train_mask_all.shape (96, 64, 64)
    test_mask_all.shape (28, 64, 64)
    train_time_all.shape (96,)
    test_time_all.shape (28,)
    """
    print("XXX:train_data_all",train_data_all.min(),train_data_all.max())
    train_data_all = Normalize(train_data_all)
    test_data_all = Normalize(test_data_all)

    train_data_all = N_S_deal(train_data_all, N_S_ratio)
    test_data_all = N_S_deal(test_data_all, N_S_ratio)
    # data_all = clou_mask('/home/haida_niejie/work/Dyn/cloud mask/small.jpg',data_all)
    train_data_all = clou_mask(mask_path,train_data_all)
    test_data_all = clou_mask(mask_path,test_data_all)
    print("Masked : train_data_all.shape", train_data_all.shape)
    print("Masked : test_data_all.shape", test_data_all.shape)


    # 读取一个图片的数据形状
    # lon_all(8640) lat_all(4320)  data_all(1,4320,8640) time(tuple1) missing_all(1,4320,8640) mask_all(4320,8640)
    # lon_all, lat_all, time, data_all, missing_all, mask_all = load_gridded_nc(filename, varname)
    data_trans = transfun[0](train_data_all)
    data_trans = transfun[0](test_data_all)


    # 初始化RMS存放所有循环的batch_RMS
    RMS = []
    R2 = []
    print('------------------i------------------')
    train_datagen, nvar, train_len, meandata = data_generator(
        train_lon_all, train_lat_all, train_time_all, train_data_all, train_missing_all,train_mask_all,
        ntime_win=ntime_win,
        jitter_std=jitter_std)
    test_datagen, nvar, test_len, test_meandata = data_generator(
        test_lon_all,test_lat_all, test_time_all, test_data_all, test_missing_all,test_mask_all,
        ntime_win=ntime_win,
        train=False)

    # print("train_datagen", train_datagen.shape)
    # print("test_datagen", test_datagen.shape)
   #meadata 没用

    print("Number of input variables: ", nvar)
    print(train_datagen)
    print(type(train_datagen))

    _, batch_RMS,batch_R2 = reconstruct(
        train_lon_all, train_lat_all, train_mask_all, meandata,test_meandata,
        train_datagen, train_len,
        test_datagen, test_len,
        outdir,N_S_ratio,corrup_rate,
        # data_epoch=i,
        # batch_size=1,
        # shuffle_buffer_size=1,
        # enc_nfilter_internal=[8, 12, 18, 27],
        transfun=transfun,
        nvar=nvar,
        **kwargs)
    RMS.append(batch_RMS)
    R2.append(batch_RMS)




"""


"""
    # print(data_all.shape[2])
    # # 现在的data是(1,720,1440),我们把它分开输入下面几行，用(1,720,200)依次送进去,即第三个维度每次去200个
    # # lat 720, lon 1440
    # # missing(1,720,1440) mask(720,1440)
    # for i in range(int(data_all.shape[2]/200)):
    #
    #     lat = lat_all[:720]
    #     lon = lon_all[i*200:(i+1)*200]
    #     data_ = data_all[:,:720,i*200:(i+1)*200]
    #     missing = missing_all[:,:720,i*200:(i+1)*200]
    #     mask = mask_all[:720,i*200:(i+1)*200]
    #
    #     print('------------------i------------------')
    #     train_datagen, nvar, train_len, meandata = data_generator(
    #         lon, lat, time, data_, missing,
    #         ntime_win=ntime_win,
    #         jitter_std=jitter_std)
    #     test_datagen, nvar, test_len, test_meandata = data_generator(
    #         lon, lat, time, data_, missing,
    #         ntime_win=ntime_win,
    #         train=False)
    #
    #     print("Number of input variables: ", nvar)
    #     print(train_datagen)
    #     print(type(train_datagen))
    #
    #     _,batch_RMS = reconstruct(
    #         lon, lat, mask, meandata,
    #         train_datagen, train_len,
    #         test_datagen, test_len,
    #         outdir,
    #         data_epoch = i,
    #         # batch_size=1,
    #         # shuffle_buffer_size=1,
    #         # enc_nfilter_internal=[8, 12, 18, 27],
    #         transfun=transfun,
    #         nvar=nvar,
    #         **kwargs)
    #     RMS.append(batch_RMS)
    # RMS = np.array(RMS)
    # RMS_mean = np.mean(RMS)
    # print(f"RMS_mean:{RMS_mean}")






def reconstruct_gridded_files(fields,outdir,
                              ntime_win = 3,
                              **kwargs):
    """
Train a neural network to reconstruct missing data_ from the NetCDF variable
`varname` in the NetCDF file `filename`. Results are saved in the output
directory `outdir`. `jitter_std` is the standard deviation of the noise to be
added to the data_ during training.
See `DINCAE.reconstruct` for other keyword arguments and
`DINCAE.load_gridded_nc` for the NetCDF format.

"""

    data_full = [None] * len(fields)
    missing = [None] * len(fields)
    jitter_std = [None] * len(fields)
    transfun = [None] * len(fields)
    varnames = [None] * len(fields)
    obs_err_std = [1] * len(fields)# value is irrelevant
    lon = []
    lat = []
    time = []

    for (i,field) in enumerate(fields):
        transfun[i] = field.get("transfun",(identity,identity))
        varnames[i] = field["varname"]

        field["lon"],field["lat"],field["time"],field["data_"],field["missing"],field["mask"] = load_gridded_nc(field["filename"],field["varname"])

        data_full[i] = transfun[i][0](field["data_"])

        print("typeof- ",type(field["data_"]))
        print("typeof ",type(data_full[i]))

        missing[i] = field["missing"]
        jitter_std[i] = field.get("jitter_std",0)

    lon = fields[0]["lon"]
    lat = fields[0]["lat"]
    time = fields[0]["time"]
    mask = fields[0]["mask"]

    ndata = len(fields)

    train_datagen,nvar,train_len,meandata = data_generator_list(
        lon,lat,time,data_full,missing,
        obs_err_std = obs_err_std,
        jitter_std = jitter_std,
        ntime_win = ntime_win,
    )
    test_datagen,nvar,test_len,test_meandata = data_generator_list(
        lon,lat,time,data_full,missing,
        obs_err_std = obs_err_std,
        ntime_win = ntime_win,
        train = False)

    print("Number of input variables: ",nvar)

    fname = reconstruct(
        lon,lat,mask,meandata,
        train_datagen,train_len,
        test_datagen,test_len,
        outdir,
        transfun = transfun[0],
        nvar = nvar,
        **kwargs)

    return fname
