import argparse
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import os
import random

import warnings

warnings.filterwarnings('ignore')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from dataset import process_dataset
from model import SSL, LogReg, MVGRL_mini, GCN
from util import set_seed
from baseline import cos_sim, cos_dist, euclidean_dis, man_dis, random_wire, corr_dis

from metric import recon_metric

import torch.nn.functional as F




parser = argparse.ArgumentParser(description='ssl')

# Random seed
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--serial', type=str, default='0', help='Exp No. for repetition')

# Data preparation
parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--partition_method', type=str, default='metis', help='adopted partition method.')
parser.add_argument('--subgraph_ratio', type=float, default=1, help='size ratio of sampled subgraph.')
parser.add_argument('--adj_init', type=str, default='cos_sim', help='adj initialize method. random, zero, cos_sim, attention, ori, full')
parser.add_argument('--adj_init_random_p', type=float, default='0.5', help='Probability for edge creation when using random')

parser.add_argument('--diffusion_method', type=str, default='ppr', help='adj diffusion method.')
parser.add_argument('--alpha1', type=float, default=0.2, help='alpha1 for ppr.')
parser.add_argument('--alpha2', type=float, default=0.4, help='alpha2 for ppr.')
parser.add_argument('--sim_k', type=int, default=5, help='5, 10, 20, 50, 100, 1000.')
parser.add_argument('--encoder', type=str, default='gat', help='encoder GNN.')

parser.add_argument('--gpu', type=int, default=1, help='GPU index. Default: -1, using cpu.')
parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of ssl.')
parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0., help='Weight decay of ssl.')
parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')
parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')

# Reconstruction
parser.add_argument('--result_mode', type=str, default='auc', help='Exp item')
parser.add_argument('--method', type=str, default='ssl', help='reconstruction method. ssl, fo, po, em')

# parser.add_argument("--feature_type", type=str, default='id', help='feature type in GVAE.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

# set seed

set_seed(args.seed)



if __name__ == '__main__':
    print(args)

    # Step 0: Train target GNN =================================================================== #
    target_graph, diff_graph_1, diff_graph_2, target_feat, target_labels, target_train_mask, target_val_mask, target_test_mask, edge_weight_1, edge_weight_2, target_n_classes = process_dataset(args.dataname, args.epsilon, args.partition_method, args.subgraph_ratio, args.adj_init, args.adj_init_random_p, args.alpha1, args.alpha2, args.sim_k)
    target_graph = target_graph.to(args.device)
    target_feat = target_feat.to(args.device)
    target_labels = target_labels.to(args.device)
    target_train_mask = target_train_mask.to(args.device)
    target_val_mask = target_val_mask.to(args.device)
    target_test_mask = target_test_mask.to(args.device)

    target_n_node = target_graph.number_of_nodes()
    target_adj = target_graph.adj().to_dense().to(args.device)
    target_adj[target_adj == 2] = 1
    target_adj_np = target_adj.cpu().numpy()

    idx_attack = np.array(random.sample(range(target_n_node), int(target_n_node)))

    if args.dataname == 'cora':
        GNN_hid = 16
    elif args.dataname == 'citeseer':
        GNN_hid = 64
    elif args.dataname == 'pubmed' or 'reddit':
        GNN_hid = 16

    if args.result_mode == 'auc' or args.method in ['po', 'em']:
        GNN = GCN(target_feat.shape[1], GNN_hid, target_n_classes)
        GNN = GNN.to(args.device)
        optimizer = torch.optim.Adam(GNN.parameters(), lr=0.01)
        best_val_acc = 0
        best_test_acc = 0
        for epoch in range(500):
            _, logits = GNN(target_graph, target_feat)
            pred = logits.argmax(1)
            loss = F.cross_entropy(logits[target_train_mask], target_labels[target_train_mask])

            # Compute accuracy on training/validation/test
            train_acc = (pred[target_train_mask] == target_labels[target_train_mask]).float().mean()
            val_acc = (pred[target_val_mask] == target_labels[target_val_mask]).float().mean()
            test_acc = (pred[target_test_mask] == target_labels[target_test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                    epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc))

        # Access posteriors
        GNN.eval()
        with torch.no_grad():
            embeds_released, logits_quried = GNN(target_graph, target_feat)





    if args.method == 'ssl':
        # ATTACKER------------------------------------!!!!!!!!!!!!!!!_---------------------------------------------------------------

        # Step 1: Prepare data =================================================================== #
        n_feat = target_feat.shape[1]

        # graph = graph.to(args.device)
        diff_graph_1 = diff_graph_1.to(args.device)
        edge_weight_1 = torch.tensor(edge_weight_1).float().to(args.device)
        diff_graph_2 = diff_graph_2.to(args.device)
        edge_weight_2 = torch.tensor(edge_weight_2).float().to(args.device)

        feat = target_feat.to(args.device)
        n_node = target_graph.number_of_nodes()
        lbl1 = torch.ones(n_node * 2)
        lbl2 = torch.zeros(n_node * 2)
        lbl = torch.cat((lbl1, lbl2))  # 5544
        lbl = lbl.to(args.device)


        # Step 2: Create model =================================================================== #
        ## Create SSL model
        model = SSL(n_feat, args.hid_dim, encoder=args.encoder)
        model = model.to(args.device)

        # Step 3: Create training components ===================================================== #
        loss_fn = nn.BCEWithLogitsLoss()
        loss_logits = nn.MSELoss()

        # Step 4: Training epochs ================================================================ #
        best = float('inf')
        cnt_wait = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

        loss_list = []

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            # Contrastive Learning
            shuf_idx = np.random.permutation(n_node)
            shuf_feat = feat[shuf_idx, :]
            shuf_feat = shuf_feat.to(args.device)
            out = model(diff_graph_1, diff_graph_2, feat, shuf_feat, edge_weight_1, edge_weight_2)

            if args.encoder == 'gat':
                out = out.squeeze()
            loss = loss_fn(out, lbl)  # Contrastive Loss
            loss.backward()
            optimizer.step()


            loss_list.append(loss.item())
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), f'save/{args.dataname}/model_{args.serial}_{args.sim_k}.pkl')
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping')
                break

        np.save(f'${args.dataname}_train_loss.npy', np.array(loss_list))


    if args.result_mode == "auc":
        # Ours
        print('-------------------------------------Our results (SSL-GSR)----------------------------------------')
        best_model = torch.load(f'save/{args.dataname}/model_{args.serial}_{args.sim_k}.pkl')
        embeds = model.get_embedding(diff_graph_1, diff_graph_2, feat, edge_weight_1, edge_weight_2)

        if args.encoder == 'gat':
            embeds = embeds.squeeze()

        embeds = embeds.cpu().numpy()

        cos_sim(args, target_adj_np, embeds, idx_attack, method='ssl', result=args.result_mode)
        cos_dist(args, target_adj_np, embeds, idx_attack, method='ssl', result=args.result_mode)
        euclidean_dis(args, target_adj_np, embeds, idx_attack, method='ssl', result=args.result_mode)
        man_dis(args, target_adj_np, embeds, idx_attack, method='ssl', result=args.result_mode)
        corr_dis(args, target_adj_np, embeds, idx_attack, method='ssl', result=args.result_mode)
        print('\n'*3)



        #--------------------Baseline-----------------------------------------------------------------------------------
        print('-------------------------------------Baselines----------------------------------------')
        target_feat_np = target_feat.detach().cpu().numpy()
        cos_sim(args, target_adj_np, target_feat_np, idx_attack, method='fo', result=args.result_mode)
        cos_dist(args, target_adj_np, target_feat_np, idx_attack, method='fo', result=args.result_mode)
        euclidean_dis(args, target_adj_np, target_feat_np, idx_attack, method='fo', result=args.result_mode)
        man_dis(args, target_adj_np, target_feat_np, idx_attack, method='fo', result=args.result_mode)
        corr_dis(args, target_adj_np, target_feat_np, idx_attack, method='fo', result=args.result_mode)
        random_wire(args, target_adj_np, prob=0.2, idx=idx_attack)
        print('\n'*3)

        print('-------------------------------------Posteriors----------------------------------------')
        posterior_np = logits_quried.detach().cpu().numpy()
        cos_sim(args, target_adj_np, posterior_np, idx_attack, method='po', result=args.result_mode)
        cos_dist(args, target_adj_np, posterior_np, idx_attack, method='po', result=args.result_mode)
        euclidean_dis(args, target_adj_np, posterior_np, idx_attack, method='po', result=args.result_mode)
        man_dis(args, target_adj_np, posterior_np, idx_attack, method='po', result=args.result_mode)
        corr_dis(args, target_adj_np, posterior_np, idx_attack, method='po', result=args.result_mode)
        print('\n'*3)

        print('-------------------------------------Embeds (Usenix 2022)----------------------------------------')
        ebeds_np = embeds_released.detach().cpu().numpy()

        cos_sim(args, target_adj_np, ebeds_np, idx_attack, method='em', result=args.result_mode)
        cos_dist(args, target_adj_np, ebeds_np, idx_attack, method='em', result=args.result_mode)
        euclidean_dis(args, target_adj_np, ebeds_np, idx_attack, method='em', result=args.result_mode)
        man_dis(args, target_adj_np, ebeds_np, idx_attack, method='em', result=args.result_mode)
        corr_dis(args, target_adj_np, ebeds_np, idx_attack, method='em', result=args.result_mode)
        print('\n'*3)

    elif args.result_mode == "util":
        # Ours
        if args.method == 'ssl':
            print('-------------------------------------Our results (SSL-GSR)----------------------------------------')
            best_model = torch.load(f'save/{args.dataname}/model_{args.serial}.pkl')
            embeds = model.get_embedding(diff_graph_1, diff_graph_2, feat, edge_weight_1, edge_weight_2)
            embeds = embeds.cpu().numpy()
            recon_graph = corr_dis(args, target_adj_np, embeds, idx_attack, method='ssl', result=args.result_mode)
        elif args.method == 'fo':
            target_feat_np = target_feat.detach().cpu().numpy()
            recon_graph = corr_dis(args, target_adj_np, target_feat_np, idx_attack, method='ssl', result=args.result_mode)
        elif args.method == 'po':
            posterior_np = logits_quried.detach().cpu().numpy()
            recon_graph = corr_dis(args, target_adj_np, posterior_np, idx_attack, method='ssl', result=args.result_mode)
        elif args.method == 'em':
            ebeds_np = embeds_released.detach().cpu().numpy()
            recon_graph = corr_dis(args, target_adj_np, ebeds_np, idx_attack, method='ssl', result=args.result_mode)

        recon_graph = recon_graph.to(args.device)
        GNN2 = GCN(target_feat.shape[1], GNN_hid, target_n_classes)
        GNN2 = GNN2.to(args.device)
        optimizer2 = torch.optim.Adam(GNN2.parameters(), lr=0.01)
        best_val_acc = 0
        best_test_acc = 0
        for epoch in range(500):
            _, logits = GNN2(recon_graph, target_feat)
            pred = logits.argmax(1)
            loss = F.cross_entropy(logits[target_train_mask], target_labels[target_train_mask])

            # Compute accuracy on training/validation/test
            train_acc = (pred[target_train_mask] == target_labels[target_train_mask]).float().mean()
            val_acc = (pred[target_val_mask] == target_labels[target_val_mask]).float().mean()
            test_acc = (pred[target_test_mask] == target_labels[target_test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            # Backward
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            if epoch % 10 == 0:
                print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                    epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc))


