#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 11/12/2022 11:34 pm
# @Author  : Wizard Chenhan Zhang
# @FileName: baseline.py
# @Software: PyCharm

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import cdist

from metric import recon_metric
import numpy as np
import time
import os
import dgl

from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample


# def cos_sim(ori_adj, feat_np, idx):
#     sim = cosine_similarity(feat_np)
#     AUC, AP = recon_metric(ori_adj, sim, idx)
#     print("cos_sim. AUC: %f AP: %f" % (AUC, AP))



def cos_sim(args, ori_adj, feat_np, idx, method='ssl', result='auc'):
    sim = cosine_similarity(feat_np)
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)


    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) > np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]

    edge_assign[pos_ind] = 1.0


    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    # print((time_end-time_start)/60)



    AUC, AP = recon_metric(ori_adj, sim_, idx)
    if method == 'ssl':
        print("SSL-GSR -- cos_sim. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'fo':
        print("Feature only -- cos_sim. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'po':
        print("Posterior -- cos_sim. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'em':
        print("Embedding -- cos_sim. AUC: %f AP: %f" % (AUC, AP))

    if result == 'auc':
        if os.path.exists(f'result/auc/{args.dataname}/{method}_cos_sim_auc.npy'):
            cos_sim_auc = np.load(f'result/auc/{args.dataname}/{method}_cos_sim_auc.npy')
        else:
            cos_sim_auc = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_cos_sim_auc.npy', cos_sim_auc)

        if os.path.exists(f'result/auc/{args.dataname}/{method}_cos_sim_ap.npy'):
            cos_sim_ap = np.load(f'result/auc/{args.dataname}/{method}_cos_sim_ap.npy')
        else:
            cos_sim_ap = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_cos_sim_ap.npy', cos_sim_ap)

        cos_sim_auc = np.append(cos_sim_auc, AUC)
        np.save(f'result/auc/{args.dataname}/{method}_cos_sim_auc.npy', cos_sim_auc)

        cos_sim_ap = np.append(cos_sim_ap, AP)
        np.save(f'result/auc/{args.dataname}/{method}_cos_sim_ap.npy', cos_sim_ap)
    if result == 'util':
        return sim_





def cos_dist(args, ori_adj, feat_np, idx, method='ssl', result='auc'):
    sim = cdist(feat_np, feat_np, 'cosine')
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)


    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) < np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]

    edge_assign[pos_ind] = 1.0




    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    # print((time_end-time_start)/60)

    np.save('ori_adj.npy', ori_adj)
    np.save(f'{method}_recon_adj.npy', sim_)

    AUC, AP = recon_metric(ori_adj, sim_, idx)
    if method == 'ssl':
        print("SSL-GSR -- cos_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'fo':
        print("Feature only -- cos_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'po':
        print("Posterior -- cos_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'em':
        print("Embedding -- cos_dis. AUC: %f AP: %f" % (AUC, AP))

    if result == 'auc':
        if os.path.exists(f'result/auc/{args.dataname}/{method}_cos_dis_auc.npy'):
            cos_dis_auc = np.load(f'result/auc/{args.dataname}/{method}_cos_dis_auc.npy')
        else:
            cos_dis_auc = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_cos_dis_auc.npy', cos_dis_auc)

        if os.path.exists(f'result/auc/{args.dataname}/{method}_cos_dis_ap.npy'):
            cos_dis_ap = np.load(f'result/auc/{args.dataname}/{method}_cos_dis_ap.npy')
        else:
            cos_dis_ap = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_cos_dis_ap.npy', cos_dis_ap)

        cos_dis_auc = np.append(cos_dis_auc, AUC)
        np.save(f'result/auc/{args.dataname}/{method}_cos_dis_auc.npy', cos_dis_auc)

        cos_dis_ap = np.append(cos_dis_ap, AP)
        np.save(f'result/auc/{args.dataname}/{method}_cos_dis_ap.npy', cos_dis_ap)
    if result == 'util':
        return sim_

def euclidean_dis(args, ori_adj, feat_np, idx, method='ssl', result='auc'):
    sim = euclidean_distances(feat_np)
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)


    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) < np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]

    edge_assign[pos_ind] = 1.0


    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    # print((time_end-time_start)/60)

    AUC, AP = recon_metric(ori_adj, sim_, idx)
    if method == 'ssl':
        print("SSL-GSR -- euclidean_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'fo':
        print("Feature only -- euclidean_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'po':
        print("Posterior -- euclidean_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'em':
        print("Embedding -- euclidean_dis. AUC: %f AP: %f" % (AUC, AP))

    if result == 'auc':
        if os.path.exists(f'result/auc/{args.dataname}/{method}_euc_dis_auc.npy'):
            euc_dis_auc = np.load(f'result/auc/{args.dataname}/{method}_euc_dis_auc.npy')
        else:
            euc_dis_auc = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_euc_dis_auc.npy', euc_dis_auc)

        if os.path.exists(f'result/auc/{args.dataname}/{method}_euc_dis_ap.npy'):
            euc_dis_ap = np.load(f'result/auc/{args.dataname}/{method}_euc_dis_ap.npy')
        else:
            euc_dis_ap = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_euc_dis_ap.npy', euc_dis_ap)

        euc_dis_auc = np.append(euc_dis_auc, AUC)
        np.save(f'result/auc/{args.dataname}/{method}_euc_dis_auc.npy', euc_dis_auc)

        euc_dis_ap = np.append(euc_dis_ap, AP)
        np.save(f'result/auc/{args.dataname}/{method}_euc_dis_ap.npy',euc_dis_ap)
    if result == 'util':
        return sim_


def man_dis(args, ori_adj, feat_np, idx, method='ssl', result='auc'):
    sim = manhattan_distances(feat_np)
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)


    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) < np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]

    edge_assign[pos_ind] = 1.0


    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    # print((time_end-time_start)/60)

    AUC, AP = recon_metric(ori_adj, sim_, idx)

    if method == 'ssl':
        print("SSL-GSR -- man_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'fo':
        print("Feature only -- man_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'po':
        print("Posterior -- man_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'em':
        print("Embedding -- man_dis. AUC: %f AP: %f" % (AUC, AP))

    if result == 'auc':
        if os.path.exists(f'result/auc/{args.dataname}/{method}_man_dis_auc.npy'):
            man_dis_auc = np.load(f'result/auc/{args.dataname}/{method}_man_dis_auc.npy')
        else:
            man_dis_auc = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_man_dis_auc.npy', man_dis_auc)

        if os.path.exists(f'result/auc/{args.dataname}/{method}_man_dis_ap.npy'):
            man_dis_ap = np.load(f'result/auc/{args.dataname}/{method}_man_dis_ap.npy')
        else:
            man_dis_ap = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_man_dis_ap.npy', man_dis_ap)

        man_dis_auc = np.append(man_dis_auc, AUC)
        np.save(f'result/auc/{args.dataname}/{method}_man_dis_auc.npy', man_dis_auc)

        man_dis_ap = np.append(man_dis_ap, AP)
        np.save(f'result/auc/{args.dataname}/{method}_man_dis_ap.npy', man_dis_ap)

    if result == 'util':
        return sim_

def corr_dis(args, ori_adj, feat_np, idx, method='ssl', result='auc'):
    sim = cdist(feat_np, feat_np, 'correlation')
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)


    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) < np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]

    edge_assign[pos_ind] = 1.0


    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    # print((time_end-time_start)/60)

    AUC, AP = recon_metric(ori_adj, sim_, idx)

    if method == 'ssl':
        print("SSL-GSR -- corr_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'fo':
        print("Feature only -- corr_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'po':
        print("Posterior -- corr_dis. AUC: %f AP: %f" % (AUC, AP))
    elif method == 'em':
        print("Embedding -- corr_dis. AUC: %f AP: %f" % (AUC, AP))

    if result == 'auc':
        if os.path.exists(f'result/auc/{args.dataname}/{method}_corr_dis_auc.npy'):
            corr_dis_auc = np.load(f'result/auc/{args.dataname}/{method}_corr_dis_auc.npy')
        else:
            corr_dis_auc = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_corr_dis_auc.npy', corr_dis_auc)

        if os.path.exists(f'result/auc/{args.dataname}/{method}_corr_dis_ap.npy'):
            corr_dis_ap = np.load(f'result/auc/{args.dataname}/{method}_corr_dis_ap.npy')
        else:
            corr_dis_ap = np.array([])
            np.save(f'result/auc/{args.dataname}/{method}_corr_dis_ap.npy', corr_dis_ap)

        corr_dis_auc = np.append(corr_dis_auc, AUC)
        np.save(f'result/auc/{args.dataname}/{method}_corr_dis_auc.npy', corr_dis_auc)

        corr_dis_ap = np.append(corr_dis_ap, AP)
        np.save(f'result/auc/{args.dataname}/{method}_corr_dis_ap.npy', corr_dis_ap)

    if result == 'util':
        recon_edges = np.nonzero(sim_)
        recon_graph = dgl.graph(recon_edges)
        return recon_graph




def random_wire(args, ori_adj, prob, idx):
    # prob_adj = np.full((ori_adj.shape[0], ori_adj.shape[0]), prob)
    # np.fill_diagonal(prob_adj, 1)

    prob_adj = np.random.uniform(low=0, high=1, size=(ori_adj.shape[0], ori_adj.shape[0]))
    prob_adj = np.tril(prob_adj) + np.tril(prob_adj, -1).T
    np.fill_diagonal(prob_adj, 1)

    AUC, AP = recon_metric(ori_adj, prob_adj, idx)
    print("random wiring with prob %f. AUC: %f AP: %f" % (prob, AUC, AP))


