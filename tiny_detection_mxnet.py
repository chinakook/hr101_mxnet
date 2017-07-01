
# coding: utf-8

# In[1]:

import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import pylab as pl
from collections import namedtuple
import time
Batch = namedtuple('Batch', ['data'])


# In[2]:

MAX_INPUT_DIM=5000.0
prob_thresh = 0.5
nms_thresh = 0.1


# In[3]:

def loadmeta(matpath):
    f = sio.loadmat(matpath)
    net = f['net']
    clusters = np.copy(net['meta'][0][0][0][0][6])
    averageImage = np.copy(net['meta'][0][0][0][0][2][0][0][2])
    averageImage = averageImage[:, np.newaxis]
    return clusters, averageImage


# In[4]:

def nms(dets, prob_thresh):
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4]  
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  

    order = scores.argsort()[::-1]  

    keep = []  
    while order.size > 0: 
        i = order[0]  
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= prob_thresh)[0]  

        order = order[inds + 1]  
    return keep


# In[5]:

clusters, averageImage = loadmeta('./hr_res101.mat')


# In[6]:

clusters_h = clusters[:,3] - clusters[:,1] + 1
clusters_w = clusters[:,2] - clusters[:,0] + 1
normal_idx = np.where(clusters[:,4] == 1)


# In[7]:

raw_img = cv2.imread('./selfie.jpg')
raw_h = raw_img.shape[0]
raw_w = raw_img.shape[1]
raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
raw_img_f = raw_img.astype(np.float32)


# In[8]:

min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx]/raw_w))), np.floor(np.log2(np.max(clusters_h[normal_idx]/raw_h))))
max_scale = min(1.0, -np.log2(max(raw_h, raw_w)/MAX_INPUT_DIM))


# In[9]:

scales_down = pl.frange(min_scale, 0, 1.)
scales_up = pl.frange(0.5, max_scale,0.5)
scales_pow = np.hstack((scales_down, scales_up))
scales = np.power(2.0, scales_pow)


# In[10]:

sym, arg_params, aux_params = mx.model.load_checkpoint('hr101',0)
all_layers = sym.get_internals()


# In[11]:

context=mx.gpu()


# In[12]:

mod = mx.mod.Module(symbol=all_layers['fusex_output'], context=context, data_names=['data'], label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=None, force_rebind=False)
mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)


# In[13]:

start = time.time()
bboxes = np.empty(shape=(0,5))
for s in scales:
    img = cv2.resize(raw_img_f, (0,0), fx = s, fy = s)
    img = np.transpose(img,(2,0,1))
    img = img - averageImage

    tids = []
    if s <= 1. :
        tids = list(range(4, 12))
    else :
        tids = list(range(4, 12)) + list(range(18, 25))
    ignoredTids = list(set(range(0,clusters.shape[0]))-set(tids))
    img_h = img.shape[1]
    img_w = img.shape[2]
    img = img[np.newaxis, :]

    mod.reshape(data_shapes=[('data', (1, 3, img_h, img_w))])
    mod.forward(Batch([mx.nd.array(img)]))
    mod.get_outputs()[0].wait_to_read()
    fusex_res = mod.get_outputs()[0]

    score_cls = mx.nd.slice_axis(fusex_res, axis=1, begin=0, end=25, name='score_cls')
    score_reg = mx.nd.slice_axis(fusex_res, axis=1, begin=25, end=None, name='score_reg')
    prob_cls = mx.nd.sigmoid(score_cls)
    
    prob_cls_np = prob_cls.asnumpy()
    prob_cls_np[0,ignoredTids,:,:] = 0.
    
    _, fc, fy, fx = np.where(prob_cls_np > prob_thresh)
    
    cy = fy * 8 - 1
    cx = fx * 8 - 1
    ch = clusters[fc, 3] - clusters[fc,1] + 1
    cw = clusters[fc, 2] - clusters[fc, 0] + 1
    
    Nt = clusters.shape[0]
    
    score_reg_np = score_reg.asnumpy()
    tx = score_reg_np[0, 0:Nt, :, :]
    ty = score_reg_np[0, Nt:2*Nt,:,:]
    tw = score_reg_np[0, 2*Nt:3*Nt,:,:]
    th = score_reg_np[0,3*Nt:4*Nt,:,:]
    
    dcx = cw * tx[fc, fy, fx]
    dcy = ch * ty[fc, fy, fx]
    rcx = cx + dcx
    rcy = cy + dcy
    rcw = cw * np.exp(tw[fc, fy, fx])
    rch = ch * np.exp(th[fc, fy, fx])
    
    score_cls_np = score_cls.asnumpy()
    scores = score_cls_np[0, fc, fy, fx]
    
    tmp_bboxes = np.vstack((rcx-rcw/2, rcy-rch/2, rcx+rcw/2,rcy+rch/2))
    tmp_bboxes = np.vstack((tmp_bboxes/s, scores))
    tmp_bboxes = tmp_bboxes.transpose()
    bboxes = np.vstack((bboxes, tmp_bboxes))


# In[14]:

print ("time", time.time()-start, "secs.")
refind_idx = nms(bboxes, nms_thresh)
refind_bboxes = bboxes[refind_idx]
refind_bboxes = refind_bboxes.astype(np.int32)
for r in refind_bboxes:
    cv2.rectangle(raw_img, (r[0],r[1]), (r[2],r[3]), (255,255,0),3)
plt.imshow(raw_img)
plt.show()


# In[15]:

bboxes.shape

