#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 9/12/2022 3:11 pm
# @Author  : Wizard Chenhan Zhang
# @FileName: metric.py
# @Software: PyCharm

from sklearn.metrics import roc_curve, auc, average_precision_score
import numpy as np
import random


def recon_metric(ori_adj, inference_adj, idx):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)

    AUC = auc(fpr, tpr)
    AP = average_precision_score(real_edge, pred_edge)
    return AUC, AP

