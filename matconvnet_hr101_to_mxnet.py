
# coding: utf-8

# In[62]:

import mxnet as mx
import numpy as np
import scipy.io as sio


# In[63]:

symbol_string = "import mxnet as mx\ndata= mx.symbol.Variable(name='data')\n"


# In[64]:

matpath='./hr_res101.mat'


# In[65]:

f = sio.loadmat(matpath)
net = f['net']


# In[66]:

data = mx.symbol.Variable(name='data')
conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
# Turn cudnn off in all batchnorm layer as the cudnn does not support eps <= 0.00001
bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1 , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
conv1_relu = mx.symbol.Activation(name='conv1_relu', data=bn_conv1 , act_type='relu')
# pad right and bottom as the origin matconvnet implementation
conv1_relu_padded = mx.symbol.pad(name='conv1_relu_padded', data=conv1_relu, mode='constant', constant_value=0, pad_width=(0,0,0,0,0,1,0,1))
# pool in matconvnet use 'valid' mode but not 'full'
pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu_padded , pooling_convention='valid', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
# another choice to deal with the matconvnet's right and bottom padding
# pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1 , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=bn2a_branch2a , act_type='relu')
res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=bn2a_branch2b , act_type='relu')
res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2a = mx.symbol.broadcast_add(name='res2a', *[bn2a_branch1,bn2a_branch2c] )
res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a , act_type='relu')
res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=bn2b_branch2a , act_type='relu')
res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=bn2b_branch2b , act_type='relu')
res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu,bn2b_branch2c] )
res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b , act_type='relu')
res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=bn2c_branch2a , act_type='relu')
res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=bn2c_branch2b , act_type='relu')
res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu,bn2c_branch2c] )
res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c , act_type='relu')
res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1 , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=bn3a_branch2a , act_type='relu')
res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=bn3a_branch2b , act_type='relu')
res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3a = mx.symbol.broadcast_add(name='res3a', *[bn3a_branch1,bn3a_branch2c] )
res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a , act_type='relu')
res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=bn3b1_branch2a , act_type='relu')
res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=bn3b1_branch2b , act_type='relu')
res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu,bn3b1_branch2c] )
res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1 , act_type='relu')
res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=bn3b2_branch2a , act_type='relu')
res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=bn3b2_branch2b , act_type='relu')
res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu,bn3b2_branch2c] )
res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2 , act_type='relu')
res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=bn3b3_branch2a , act_type='relu')
res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=bn3b3_branch2b , act_type='relu')
res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu,bn3b3_branch2c] )
res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3 , act_type='relu')
res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1 , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=bn4a_branch2a , act_type='relu')
res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=bn4a_branch2b , act_type='relu')
res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4a = mx.symbol.broadcast_add(name='res4a', *[bn4a_branch1,bn4a_branch2c] )
res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a , act_type='relu')
res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=bn4b1_branch2a , act_type='relu')
res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=bn4b1_branch2b , act_type='relu')
res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu,bn4b1_branch2c] )
res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1 , act_type='relu')
res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=bn4b2_branch2a , act_type='relu')
res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=bn4b2_branch2b , act_type='relu')
res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu,bn4b2_branch2c] )
res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2 , act_type='relu')
res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=bn4b3_branch2a , act_type='relu')
res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=bn4b3_branch2b , act_type='relu')
res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu,bn4b3_branch2c] )
res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3 , act_type='relu')
res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=bn4b4_branch2a , act_type='relu')
res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=bn4b4_branch2b , act_type='relu')
res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu,bn4b4_branch2c] )
res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4 , act_type='relu')
res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=bn4b5_branch2a , act_type='relu')
res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=bn4b5_branch2b , act_type='relu')
res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu,bn4b5_branch2c] )
res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5 , act_type='relu')
res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=bn4b6_branch2a , act_type='relu')
res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=bn4b6_branch2b , act_type='relu')
res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu,bn4b6_branch2c] )
res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6 , act_type='relu')
res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=bn4b7_branch2a , act_type='relu')
res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=bn4b7_branch2b , act_type='relu')
res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu,bn4b7_branch2c] )
res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7 , act_type='relu')
res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=bn4b8_branch2a , act_type='relu')
res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=bn4b8_branch2b , act_type='relu')
res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu,bn4b8_branch2c] )
res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8 , act_type='relu')
res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=bn4b9_branch2a , act_type='relu')
res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=bn4b9_branch2b , act_type='relu')
res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu,bn4b9_branch2c] )
res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9 , act_type='relu')
res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=bn4b10_branch2a , act_type='relu')
res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=bn4b10_branch2b , act_type='relu')
res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu,bn4b10_branch2c] )
res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10 , act_type='relu')
res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=bn4b11_branch2a , act_type='relu')
res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=bn4b11_branch2b , act_type='relu')
res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu,bn4b11_branch2c] )
res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11 , act_type='relu')
res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=bn4b12_branch2a , act_type='relu')
res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=bn4b12_branch2b , act_type='relu')
res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu,bn4b12_branch2c] )
res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12 , act_type='relu')
res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=bn4b13_branch2a , act_type='relu')
res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=bn4b13_branch2b , act_type='relu')
res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu,bn4b13_branch2c] )
res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13 , act_type='relu')
res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=bn4b14_branch2a , act_type='relu')
res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=bn4b14_branch2b , act_type='relu')
res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu,bn4b14_branch2c] )
res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14 , act_type='relu')
res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=bn4b15_branch2a , act_type='relu')
res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=bn4b15_branch2b , act_type='relu')
res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu,bn4b15_branch2c] )
res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15 , act_type='relu')
res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=bn4b16_branch2a , act_type='relu')
res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=bn4b16_branch2b , act_type='relu')
res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu,bn4b16_branch2c] )
res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16 , act_type='relu')
res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=bn4b17_branch2a , act_type='relu')
res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=bn4b17_branch2b , act_type='relu')
res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu,bn4b17_branch2c] )
res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17 , act_type='relu')
res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=bn4b18_branch2a , act_type='relu')
res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=bn4b18_branch2b , act_type='relu')
res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu,bn4b18_branch2c] )
res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18 , act_type='relu')
res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=bn4b19_branch2a , act_type='relu')
res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=bn4b19_branch2b , act_type='relu')
res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu,bn4b19_branch2c] )
res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19 , act_type='relu')
res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=bn4b20_branch2a , act_type='relu')
res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=bn4b20_branch2b , act_type='relu')
res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu,bn4b20_branch2c] )
res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20 , act_type='relu')
res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=bn4b21_branch2a , act_type='relu')
res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=bn4b21_branch2b , act_type='relu')
res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu,bn4b21_branch2c] )
res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21 , act_type='relu')
res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=bn4b22_branch2a , act_type='relu')
res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=bn4b22_branch2b , act_type='relu')
res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c , use_global_stats=True, fix_gamma=False, eps=0.00001, cudnn_off=True)
res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu,bn4b22_branch2c] )
res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22 , act_type='relu')
score_res4 = mx.symbol.Convolution(name='score_res4', data=res4b22_relu , num_filter=125, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
score4 = mx.symbol.Deconvolution(name='score4', data=score_res4 , num_filter=125, pad=(0, 0), kernel=(4,4), stride=(2,2), no_bias=True)
score_res3 = mx.symbol.Convolution(name='score_res3', data=res3b3_relu , num_filter=125, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
# As the convolution block make input padding and output downsampling, the deconvolution block should make input upsampling and OUTPUT CROPPING.
# It's tricky to crop the deconvolution result with 'slice' op, same to the crop param [1,2,1,2] of ConvTranspose in matconvnet.
score4_sliced = mx.symbol.slice(name='score4_sliced', data=score4, begin=(0,0,1,1), end=(None,None,-2,-2))
crop = mx.symbol.Crop(name='crop', *[score_res3, score4_sliced] , center_crop=True)
fusex = mx.symbol.broadcast_add(name='fusex', *[score4_sliced,crop] )


# In[67]:

arg_shapes, _, aux_shapes = fusex.infer_shape(data=(1,3,224,224))
arg_names = fusex.list_arguments()
aux_names = fusex.list_auxiliary_states()
arg_shape_dic = dict(zip(arg_names, arg_shapes))
aux_shape_dic = dict(zip(aux_names, aux_shapes))
arg_params = {}
aux_params = {}


# In[70]:

layers = net['layers'][0][0][0]
mat_params = net['params'][0][0][0]
mat_params_dict = {}
for p in mat_params:
    mat_params_dict[p[0][0]] = p[1]


# In[124]:



for k, layer in enumerate(layers):
    type_string = ''
    param_string = ''
    
    layer_name = layer[0][0]
    layer_type = layer[1][0]
    
    layer_inputs = []
    layer_outputs = []
    layer_params = []
    
    layer_inputs_count=layer[2][0].shape[0]
    for i in range(layer_inputs_count):
        layer_inputs.append(layer[2][0][i][0])
    
    layer_outputs_count=layer[3][0].shape[0]
    for i in range(layer_outputs_count):
        layer_outputs.append(layer[3][0][i][0])
    
    if layer[4].shape[0] > 0:
        layer_params_count = layer[4][0].shape[0]
        for i in range(layer_params_count):
            layer_params.append(layer[4][0][i][0])
    
    if layer_type == u'dagnn.Conv':
        nchw = layer[5][0][0][0][0]
        hasBias = layer[5][0][0][1][0][0]
        pad = layer[5][0][0][3][0]
        stride = layer[5][0][0][4][0]
        dilate = layer[5][0][0][5][0]
        type_string = 'mx.symbol.Convolution'
        wmat = mat_params_dict[layer_name+'_filter']
        wmat = np.transpose(wmat, [3,2,0,1]) # matlab array is (h w c n) so need to swap axes
        arg_params[layer_name+'_weight'] = mx.nd.array(wmat)
        if hasBias:
            bias = mat_params_dict[layer_name+'_bias'][0]
            arg_params[layer_name+'_bias'] = mx.nd.array(bias)
    elif layer_type == u'dagnn.BatchNorm':
        epslion = layer[5][0][0][1][0][0]
        type_string = 'mx.symbol.BatchNorm'
        gamma = mat_params_dict[layer_name+'_mult'][:,0]
        beta = mat_params_dict[layer_name+'_bias'][:,0]
        moments = mat_params_dict[layer_name+'_moments']
        moving_mean = moments[:,0]
        moving_var = moments[:,1] * moments[:,1] - epslion
        arg_params[layer_name+'_gamma'] = mx.nd.array(gamma)
        arg_params[layer_name+'_beta'] = mx.nd.array(beta)
        aux_params[layer_name+'_moving_mean'] = mx.nd.array(moving_mean)
        aux_params[layer_name+'_moving_var'] = mx.nd.array(moving_var)
    elif layer_type == u'dagnn.ConvTranspose':
        nchw = layer[5][0][0][0][0]
        hasBias = layer[5][0][0][1][0][0]
        upsample = layer[5][0][0][2][0]
        crop = layer[5][0][0][3][0]
        type_string = 'mx.symbol.Deconvolution'
        wmat = mat_params_dict[layer_name+'f']
        wmat = np.transpose(wmat, [3,2,0,1]) # matlab array is (h w c n) so need to swap axes
        arg_params[layer_name+'_weight']=mx.nd.array(wmat)
    elif layer_type == u'dagnn.Pooling':
        mathod = layer[5][0][0][0][0]
        poolSize = layer[5][0][0][1][0]
        pad = layer[5][0][0][3][0]
        stride = layer[5][0][0][4][0]
        type_string = 'mx.symbol.Pooling'
        param_string = "pooling_convention='full', "
        param_string += "pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d)" % (
            pad[0], pad[2], poolSize[0], poolSize[1],
            stride[0], stride[1])
    elif layer_type == u'dagnn.ReLU':
        type_string = 'mx.symbol.Activation'
        param_string = "act_type='relu'"
    elif layer_type == u'dagnn.Sum':
        type_string = 'mx.symbol.broadcast_add'
        param_string = ""
        pass
    else:
        pass


# In[126]:

fusex.save('hr101-symbol.json')


# In[127]:

model = mx.mod.Module(symbol=fusex, data_names=['data'], label_names=None)
model.bind(data_shapes=[('data', (1, 3, 224, 224))])
model.init_params(arg_params=arg_params, aux_params=aux_params)
model.save_checkpoint('hr101', 0)
