import logging
import mxnet as mx
import numpy as np
import cv2 as cv
import pickle
import os, sys
from collections import namedtuple
import time

Batch = namedtuple('Batch', ['data'])

class TinyFacesDetector:
    def __init__(self, model_root='./', prob_thresh=0.5, gpu_idx=0, cudnn_autotune=False, verbose=False):
        if verbose:
            logging.basicConfig(level = logging.INFO, format = '[INFO]:  %(message)s', handlers = [logging.StreamHandler()])
        logging.info('Loading detector...')
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']=str(int(cudnn_autotune))
        self.MAX_INPUT_DIM=5000.0
        self.prob_thresh = prob_thresh
        self.nms_thresh = 0.1

        sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_root, 'hr101'),0)
        all_layers = sym.get_internals()

        meta_file = open(os.path.join(model_root, 'meta.pkl'), 'rb')
        self.clusters = pickle.load(meta_file)
        self.averageImage = pickle.load(meta_file)
        meta_file.close()
        self.clusters_h = self.clusters[:,3] - self.clusters[:,1] + 1
        self.clusters_w = self.clusters[:,2] - self.clusters[:,0] + 1
        self.normal_idx = np.where(self.clusters[:,4] == 1)

        context=mx.gpu(gpu_idx)
        self.mod = mx.mod.Module(symbol=all_layers['fusex_output'], context=context, data_names=['data'], label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=None, force_rebind=False)
        self.mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)
        logging.info('Done loading.')

    @staticmethod
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

    def detect(self, img):
        raw_img = img
        raw_h = raw_img.shape[0]
        raw_w = raw_img.shape[1]
        raw_img = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)
        raw_img_f = raw_img.astype(np.float32)
        min_scale = min(np.floor(np.log2(np.max(self.clusters_w[self.normal_idx]/raw_w))), np.floor(np.log2(np.max(self.clusters_h[self.normal_idx]/raw_h))))
        max_scale = min(1.0, -np.log2(max(raw_h, raw_w)/self.MAX_INPUT_DIM))
        scales_down = np.arange(min_scale, 0+0.0001, 1.)
        scales_up = np.arange(0.5, max_scale+0.0001, 0.5)
        scales_pow = np.hstack((scales_down, scales_up))
        scales = np.power(2.0, scales_pow)

        start = time.time()
        bboxes = np.empty(shape=(0,5))
        for s in scales[::-1]:
            img = cv.resize(raw_img_f, (0,0), fx = s, fy = s)
            img = np.transpose(img,(2,0,1))
            img = img - self.averageImage

            tids = []
            if s <= 1. :
                tids = list(range(4, 12))
            else :
                tids = list(range(4, 12)) + list(range(18, 25))
            ignoredTids = list(set(range(0,self.clusters.shape[0]))-set(tids))
            img_h = img.shape[1]
            img_w = img.shape[2]
            img = img[np.newaxis, :]

            self.mod.reshape(data_shapes=[('data', (1, 3, img_h, img_w))])
            self.mod.forward(Batch([mx.nd.array(img)]))
            self.mod.get_outputs()[0].wait_to_read()
            fusex_res = self.mod.get_outputs()[0]

            score_cls = mx.nd.slice_axis(fusex_res, axis=1, begin=0, end=25, name='score_cls')
            score_reg = mx.nd.slice_axis(fusex_res, axis=1, begin=25, end=None, name='score_reg')
            prob_cls = mx.nd.sigmoid(score_cls)

            prob_cls_np = prob_cls.asnumpy()
            prob_cls_np[0,ignoredTids,:,:] = 0.

            _, fc, fy, fx = np.where(prob_cls_np > self.prob_thresh)

            cy = fy * 8 - 1
            cx = fx * 8 - 1
            ch = self.clusters[fc, 3] - self.clusters[fc,1] + 1
            cw = self.clusters[fc, 2] - self.clusters[fc, 0] + 1

            Nt = self.clusters.shape[0]

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

        logging.info("Detection time: " + str(time.time()-start) + "secs.")
        refind_idx = self.nms(bboxes, self.nms_thresh)
        refind_bboxes = bboxes[refind_idx]
        refind_bboxes = refind_bboxes.astype(np.int32)

        return refind_bboxes
