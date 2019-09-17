'''
Created on 2019年9月17日

@author: jinglingzhiyu
'''
import numpy as np

class CocoMetricPose(object):
    def __init__(self):
        self.coco_sigma = self.CocoPoseSigma()
    def ComputeIous(self, gt_kpts, pred_kpts, numbers, bboxs, areas, sigmas=None):
        if len(gt_kpts) == 0 or len(pred_kpts) == 0:
            return []
        if sigmas is None:
            sigmas = self.coco_sigma
        sigma = (sigmas * 2) ** 2
        ious = np.zeros((len(gt_kpts), len(pred_kpts)))
        for i in range(len(gt_kpts)):
            g = gt_kpts[i]
            xg = g[:,0]; yg = g[:,1]; vg = g[:,2]
            k1 = np.count_nonzero(vg > 0)
            bb = bboxs[i]
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for j in range(len(pred_kpts)):
                d = pred_kpts[j]
                xd = d[:,0]; yd = d[:,1]
                if k1>0:
                    dx = xd - xg
                    dy = yd - yg
                else:
                    z = np.zeros((len(sigmas)))
                    dx = np.max((z, x0-xd), axis=0) + np.max((z, xd-x1), axis=0)
                    dy = np.max((z, y0-yd), axis=0) + np.max((z, yd-y1), axis=0)
                e = (dx ** 2 + dy ** 2) / 2 / sigma / (areas[i]+np.spacing(1))
                if k1 > 0:
                    e = e[vg > 0]
                ious[i,j] = np.mean(np.exp(-e))
        return ious
    def ComputeOks(self, gt_kpts, pred_kpts, numbers, bboxs, areas, sigmas=None):
        ious = self.ComputeIous(gt_kpts, pred_kpts, numbers, bboxs, areas)
        if(len(ious) == 0):
            return ious
        iouk = np.array([np.max(ious)]) if len(ious)==1 else np.max(ious, axis=1)
        return iouk
    def CocoPoseSigma(self):
        return np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0