import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GATConv

from dgl.nn.pytorch.glob import AvgPooling
from dgl.ops import segment
import dgl

class GCN(nn.Module):
    def __init__(self, in_dim, h_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, h_dim)
        self.conv2 = GraphConv(h_dim, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h1 = F.relu(h)

        h2 = self.conv2(g, h1)
        return h1, h2


class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)

        logits = th.cat((sc_1, sc_2, sc_3, sc_4))

        return logits

class SSL(nn.Module):

    def __init__(self, in_dim, out_dim, encoder='gcn'):
        super(SSL, self).__init__()

        self.encoder = encoder

        if self.encoder == 'gcn':
            self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
            self.encoder2 = GraphConv(in_dim, out_dim, norm='none', bias=True, activation=nn.PReLU())

        elif self.encoder == 'gat':
            self.encoder1 = GATConv(in_dim, out_dim, num_heads=1, bias=True, activation=nn.PReLU())
            self.encoder2 = GATConv(in_dim, out_dim, num_heads=1, bias=True, activation=nn.PReLU())

        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, diff_graph_1, diff_graph_2, feat, edge_weight_1, edge_weight_2):
        if self.encoder == 'gcn':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)
        elif self.encoder == 'gat':
            h1 = self.encoder1(diff_graph_1, feat)
            h2 = self.encoder2(diff_graph_2, feat)
        return (h1 + h2).detach()


    def forward(self, diff_graph_1, diff_graph_2, feat, shuf_feat, edge_weight_1, edge_weight_2):
        # GCN
        if self.encoder == 'gcn':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)

            h3 = self.encoder1(diff_graph_1, shuf_feat, edge_weight=edge_weight_1)
            h4 = self.encoder2(diff_graph_2, shuf_feat, edge_weight=edge_weight_2)

        if self.encoder == 'gat':
            h1 = self.encoder1(diff_graph_1, feat)
            h2 = self.encoder2(diff_graph_2, feat)

            h3 = self.encoder1(diff_graph_1, shuf_feat)
            h4 = self.encoder2(diff_graph_2, shuf_feat)

        c1 = self.act_fn(self.pooling(diff_graph_1, h1))
        c2 = self.act_fn(self.pooling(diff_graph_2, h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out

class MVGRL_ori(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MVGRL, self).__init__()

        self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
        self.encoder2 = GraphConv(in_dim, out_dim, norm='none', bias=True, activation=nn.PReLU())
        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, graph, diff_graph, feat, edge_weight):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        return (h1 + h2).detach()

    def forward(self, graph, diff_graph, feat, shuf_feat, edge_weight):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        h3 = self.encoder1(graph, shuf_feat)
        h4 = self.encoder2(diff_graph, shuf_feat, edge_weight=edge_weight)

        c1 = self.act_fn(self.pooling(graph, h1))
        c2 = self.act_fn(self.pooling(graph, h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out


class MVGRL_mini(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MVGRL_mini, self).__init__()

        self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
        self.encoder2 = GraphConv(in_dim, out_dim, norm='none', bias=True, activation=nn.PReLU())
        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, g1, g2, feat, edge_weight):
        h1 = self.encoder1(g1, feat)
        h2 = self.encoder2(g2, feat, edge_weight=edge_weight)

        return (h1 + h2).detach()

    def forward(self, g1, g2, feat, shuf_feat, edge_weight):
        h1 = self.encoder1(g1, feat)
        h2 = self.encoder2(g2, feat, edge_weight=edge_weight)

        h3 = self.encoder1(g1, shuf_feat)
        h4 = self.encoder2(g2, shuf_feat, edge_weight=edge_weight)

        # c1 = self.act_fn(h1.mean(dim=0))
        c1 = self.act_fn(self.pooling(g1, h1))
        # c2 = self.act_fn(h2.mean(dim=0))
        c2 = self.act_fn(self.pooling(g1, h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out