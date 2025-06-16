# -*- coding: utf-8 -*-
# @Time : 2025/2/10 20:07
# @Author : wwj

from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from statistics import mean
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
device = torch.device('cuda')

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def ortho_norm(weight):
    wtw = torch.mm(weight.t(), weight) + 1e-4 * torch.eye(weight.shape[1]).to(weight.device)
    L = torch.linalg.cholesky(wtw)
    weight_ortho = torch.mm(weight, L.inverse().t())
    return weight_ortho
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False, activation=torch.tanh):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

        self.weight = glorot_init(in_features, out_features)
        self.ortho_weight = torch.zeros_like(self.weight)
        self.activation = activation

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        self.ortho_weight = ortho_norm(self.weight)
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        # output = theta*torch.mm(support, self.weight)+(1-theta)*r
        output = theta * torch.mm(support, self.ortho_weight) + (1 - theta) * r
        if self.residual:
            output = output+input
        return self.activation(output)

# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         input = torch.tensor(input, dtype=torch.float32).to(device)
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

class GCN(nn.Module):
    #初始化操作
    def __init__(self, nfeat, nhidden, nclass, dropout):
        super(GCN, self).__init__()
        for ij in range(len(nfeat)):
            nfeat_j = nfeat[ij]
        self.gc1 = GraphConvolution(nfeat_j, nhidden)
        self.gc2 = GraphConvolution(nhidden, nclass)
        self.dropout = dropout

    #前向传播
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class MAUGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, phi, nclass, dropout, lamda, alpha, variant):
        super(MAUGCN, self).__init__()

        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant))
        self.layer = nlayers
        self.fcs = nn.ModuleList()
        for ij in range(len(nfeat)):
            nfeat_j = nfeat[ij]
            self.fcs.append(nn.Linear(nfeat_j, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))

        # self.catt= SelfAttention(nhidden)


        # self.att1 = GraphAttentionLayer(nhidden, nhidden,dropout,alpha,concat=True)
        # self.att2 = SelfAttentionWide(nhidden, heads=8, mask=False)

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        # self.params3 = list(self.catt.parameters())

        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.hidden = nhidden
        self.phi = phi
    def forward(self, x, adj, nfeat_sum):
        layer_fcs = []
        outputtsum = 0
        outputs = []
        outputts_out = []
        output_allcons = []
        weights = []
        for k in range(len(adj)):
            oth = len(adj)+1
            adjj = adj[k]
            input = x[k]
            nfeat = nfeat_sum[k]
            nhidden = self.hidden
            phi =self.phi
            # x = torch.log(torch.from_numpy(np.array(x.cpu(),np.float)))

            # block = FCAttention( nfeat, hidden=self.hidden)
            # layer_inner = block(input)

            layer_inner = F.dropout(input, self.dropout, training=self.training)
            layer_inner = self.act_fn(self.fcs[k](layer_inner))
            layer_fcs.append(layer_inner)
            output_cons = []
            for i, con in enumerate(self.convs):
                # content = self.att1(layer_inner,adjj)
                layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                #1

                # layer_inner = self.act_fn(con(layer_inner, adjj, layer_fcs[k], self.lamda, self.alpha, i + 1))#GCNII
                # layer_inner = layer_inner*weights[0]
                # block  = FCAttention(nhidden, phi)
                # layer_inner = block(layer_inner)
                if k == 0 :
                    block = FCAttention(nhidden)
                    layer_inner, weight = block(layer_inner)
                    weights.append(weight)
                    layer_inner = self.act_fn(con(layer_inner, adjj, layer_fcs[k], self.lamda, self.alpha, i + 1))
                else:
                    layer_inner = self.act_fn(con(layer_inner, adjj, layer_fcs[k], self.lamda, self.alpha, i + 1))  # GCNII
                    layer_inner = layer_inner * weights[i]
                # else:
                #     layer_inner, weight = block(layer_inner, weight[0])
                output_cons.append(layer_inner)
                output_allcons.append(layer_inner)
            outputts= F.dropout(layer_inner, self.dropout, training=self.training)
            outputs.append(outputts)
        for kj in range(len(adj)):
            layer_output = self.fcs[-1](outputs[kj])
            outputtsum = outputtsum +layer_output
            outputts_out.append(F.log_softmax(layer_output,dim=1))

        outputsmean = torch.mean(torch.stack(outputts_out[0:len(adj)]), dim=0, keepdim=True)
        return F.log_softmax(outputtsum,dim=1),outputsmean.squeeze(0),outputts_out

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor + fea2 * (1 - mix_factor)
        return out


class FCAttention(nn.Module):
    def __init__(self, nfeat , b=1, gamma=2):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 保持池化层以兼容四维输入
        t = int(abs((math.log(nfeat, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.fc = nn.Conv2d(nfeat, nfeat, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()
        self.mix_block = nn.Sigmoid()

    def forward(self, input ):
        # 将二维输入转换为四维（添加H和W维度为1）
        input_4d = input.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        x = self.avg_pool(input_4d)
        self.conv1 = self.conv1.to(device)
        self.fc = self.fc.to(device)
        self.mix = self.mix.to(device)
        # 处理x1
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        # 处理x2
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)

        # 计算out1
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)

        # 计算out2
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)

        # 融合out1和out2
        out = self.mix(out1, out2)

        # 最终卷积和激活
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        # out = self.phi * out + (1 - self.phi) * weight
        # 压缩回二维 (B, C)
        out = out.squeeze(-1).squeeze(-1)

        return input *  out, out
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, input_a, input_b):
        # Compute query, key, and value
        query = self.query(input_a)
        key = self.key(input_a)
        value = self.value(input_b)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1)).float())
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, value)

        # Sum along the sequence dimension
        return attended_values
