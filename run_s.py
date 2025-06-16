# -*- coding: utf-8 -*-
# @Time : 2025/1/11
# @Author : Wang Wei Jun

from __future__ import division
from __future__ import print_function
from sklearn.metrics import f1_score
from  Kmeans import kmeans_adj
import random
import argparse
from scipy.signal import savgol_filter
import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from scipy.special import gamma
import scipy.sparse as sp

import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import *
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
# 计算两点距离
from matplotlib.widgets import RectangleSelector
from sklearn.cluster import k_means
from model import MAUGCN
from time import perf_counter, time
from LoderData import dataprocess
import os
os.environ['OMP_NUM_THREADS'] = '1'
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd3', type=float, default=0.005, help='weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='hidden dimensions.')
parser.add_argument('--hidden1', type=int, default=32, help='hidden dimensions.')
parser.add_argument('--phi', type=int, default=0.9, help='weight factor.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--lamda1',nargs='+', type=float, default=[1], help='weight for the attention.')
# parser.add_argument('--lamda1', type=float, default=0.5,help='weight')
# parser.add_argument('--lamda1',nargs='+', type=float, default=[0.01,0.6,0.7,0.9,1], help='weight for the attention.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
parser.add_argument('--wd11', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd22', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
parser.add_argument('--alpha1', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--batch_size2', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=10, help='batchsize for train')
parser.add_argument('--test_gap', type=int, default=10,help='the train epochs between two test')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--dataset-name", nargs="?", default="ALOI")
parser.add_argument("--k", type=int, default=15, help="k of KNN graph.")
parser.add_argument("--beta", type=float, default=1.0, help="beta. Default is 1.0")
parser.add_argument("--rho", type=float, default=0.05, help="rho. Default is 0.05")
parser.add_argument("--ratio", type=float, default=0.2, help="Ratio of labeled samples")
parser.add_argument("--sf_seed", type=int, default=2042, help="Random seed for train-test split. Default is 42.")
args = parser.parse_args()
args, unknown = parser.parse_known_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
#cudaid = "cuda:"+str(args.dev)
#device = torch.device(cudaid)

label = 5

def visual1(output,train_labels,epoch,R,name):
    output = output.to('cpu').detach().numpy()
    train_labels = train_labels.to('cpu').detach().numpy()
    output = TSNE(n_components=2).fit_transform(output)#ͶӰ�ڶ�ά��
    figure = plt.figure(figsize=(5, 5))#����ͼ�񳤶������

    color_idx = {}
    for i in range(output.shape[0]):#����ÿ���ڵ�
        color_idx.setdefault(train_labels[i], [])
        color_idx[train_labels[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(output[idx, 0], output[idx, 1], label=c, s=8)#����Ӧλ�õĵ㻭��
    plt.legend()
    plt.savefig('./fig/tsne/'+str(R)+name+'_'+str(epoch)+'.png', dpi=1000)

def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro')
    return ACC

def f1_score(output, labels):
    preds = output.max(1)[1].type_as(labels)  # ��ȡԤ������
    tp = (preds * labels).sum().to(torch.float32)  # ������������
    fp = ((preds == 1) & (labels == 0)).sum().to(torch.float32)  # �����������
    fn = ((preds == 0) & (labels == 1)).sum().to(torch.float32)  # ����ٸ�����

    precision = tp / (tp + fp + 1e-12)  # ���㾫ȷ�ʣ���ֹ��ĸΪ0��
    recall = tp / (tp + fn + 1e-12)  # �����ٻ��ʣ���ֹ��ĸΪ0��

    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)  # ����F1��������ֹ��ĸΪ0��

    return precision,recall,f1.item()  # ����F1����ֵ

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def draw_point(data):
    N = data.shape[0]
    plt.figure()
    plt.axis()
    for i in range(N):
        plt.scatter(data[i][0],data[i][1],s=16.,c='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('origin graph')
    plt.show()
def get_avg_n_div_maxr(gb_list):
    '''
    用 n / maxr 当做判定噪声球的条件，用来过滤噪声粒球
    Args:
        gb_list:

    Returns:
    '''
    avg_n_div_maxr = sum([len(x) / get_radius(x)  for x in gb_list]) / len(gb_list)
    return avg_n_div_maxr
def get_S_matrix_drr_xia(gb_list):
    '''
    得到相似度矩阵，距离越小相似度越大
    Args:
        gb_list:

    Returns:

    '''

    means = [np.mean(gb, axis=0) for gb in gb_list]
    n = len(gb_list)
    r_list = []
    for idx,gb in enumerate(gb_list):
        points = gb
        centroid = means[idx]

        distances = np.linalg.norm(points - centroid, axis=1)

        max_distance = np.max(distances)
        r_list.append(max_distance)
    rs = np.array(r_list)
    sub1 = np.array([np.repeat(element, n) for element in rs])
    sub2 = sub1.T
    means = np.array(means)
    Means = np.linalg.norm(means[:, np.newaxis] - means, axis=2)
    d_matrix = Means - sub1 - sub2
    min_d = np.min(d_matrix)
    if min_d < 0:
        min_d = min_d * (-1)
        d_matrix += min_d * 2
    np.fill_diagonal(d_matrix, 0)

    raw_gb_gb_d_matrix = 1 / d_matrix
    raw_gb_gb_d_matrix[np.isinf(raw_gb_gb_d_matrix)] = 0
    return raw_gb_gb_d_matrix

class GB_senior():

    S = None
    def __init__(self,idx_list):
        self.idx_list = idx_list

    @staticmethod
    def get_gb_gb_d_senior(gb1,gb2):
        max_s = 0
        for g1 in gb1.idx_list:
            for g2 in gb2.idx_list:
                max_s = max(GB_senior.S[g1,g2],max_s)
        return max_s
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY
def merge_sets(sets):
    '''
    合并有相同元素的集合
    Args:
        sets:

    Returns:

    '''
    uf = UnionFind()
    for set_ in sets:
        it = iter(set_)
        first_elem = next(it)
        for elem in it:
            uf.union(first_elem, elem)

    result = {}
    for set_ in sets:
        for elem in set_:
            root = uf.find(elem)
            if root not in result:
                result[root] = set()
            result[root].add(elem)

    return list(result.values())

def divide_ball_GBC_y(X,K,detaile = False):
    '''
    先根据GBC中的粒球划分生成粒球，然后按照粒球的最近邻优先的距离度量连接粒球。
    Args:
        X: 数据
        K: 聚类簇数
        detaile: 聚类细节
    Returns:
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaler = scaler.fit_transform(X)
    detaile_list = []

    minimum_ball = 2  # 一般为2，重叠时调大
    percent_avg = 0.2  # 越大忽略的越多

    # start_time = datetime.datetime.now()
    gb_list = get_gb_division_x(X, False)
    # end_time = datetime.datetime.now()
    # consume_time = (end_time - start_time)
    # print('split consume time is-----', consume_time)

    gb_list = [x for x in gb_list if len(x) != 0]
    print("gb_list len:", len(gb_list))
    return gb_list


def get_gb_division_x(data, plt_flag=False):

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    gb_list_not_temp = []
    k1 = int(np.sqrt(len(data)))
    gb_list_temp = divide_gb_k(data, k1)  # 先粗划分为根号n个粒球


    # gb_list_temp, gb_list_not_temp = division_central_consistency_strong(gb_list_temp, gb_list_not_temp)
    # gb_list_temp = gb_list_temp + gb_list_not_temp
    gb_list_not_temp = []

    i = 0
    # 根据中心一致性进行粒球细分
    while 1:
        i += 1
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = division_central_consistency(gb_list_temp, gb_list_not_temp)
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)

        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break
    count = 0
    for gb in gb_list_temp:
        if len(gb) == 1:
            pass
        else:
            count += 1
    gb_list_temp = [x for x in gb_list_temp if len(x) != 0]

    gb_list_temp = de_sparse(gb_list_temp)
    return gb_list_temp

def divide_gb_k(data, k):
    kmeans = KMeans(n_clusters=k, random_state=5)
    kmeans.fit(data)
    labels = kmeans.labels_
    gb_list_temp = []
    for idx in range(k):
        cluster1 = [data[i].tolist() for i in range(len(data)) if labels[i] == idx]
        gb_list_temp.append(np.array(cluster1))
    return gb_list_temp

def division_central_consistency_strong(gb_list,gb_list_not):
    '''
    强中心一致性划分
    Args:
        gb_list:
        gb_list_not:

    Returns:

    '''
    gb_list_new = []
    methods = '2-means'

    for gb in gb_list:
        if len(gb) > 1:
            ball_1, ball_2 = spilt_ball_k_means(gb, 2, methods)
            ccp, ccp_flag = get_ccp_strong(gb)
            t1 = ccp_flag
            t4 = len(ball_2) > 2 and len(ball_1) > 2
            if t1 and t4:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)

    return gb_list_new,gb_list_not


def spilt_ball_k_means(data, n, methods):
    if methods == '2-means':

        kmeans = KMeans(n_clusters=n, random_state=0, n_init=3, max_iter=2)
        kmeans.fit(data)
        labels = kmeans.labels_
        cluster1 = [data[i].tolist() for i in range(len(data)) if labels[i] == 0]
        cluster2 = [data[i].tolist() for i in range(len(data)) if labels[i] == 1]
        ball1 = np.array(cluster1)
        ball2 = np.array(cluster2)
        return [ball1, ball2]

    elif methods == 'k-means':

        kmeans = KMeans(n_clusters=n, random_state=1)
        kmeans.fit(data)

        labels = kmeans.labels_

        clusters = [[] for _ in range(n)]
        for i in range(len(data)):
            for cluster_index in range(n):
                if labels[i] == cluster_index:
                    clusters[cluster_index].append(data[i].tolist())

        balls = [np.array(cluster) for cluster in clusters]

        return balls
    else:
        pass
def get_ccp_strong(gb):
    '''
    强中心一致性，最大半径的四分之一半径内密度占最大半径内密度的比值
    Args:
        gb:

    Returns:

    '''
    num = len(gb)
    if num == 0:
        return 0, False
    dimension = len(gb[0])
    center = gb.mean(axis=0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5

    max_radius = np.max(distances)
    quarter_radius = max_radius / 4
    points_inside_quarter_radius = np.sum(distances <= quarter_radius)
    density_inside_quarter_radius = points_inside_quarter_radius / (quarter_radius ** dimension)
    density_max_radius = num / (max_radius ** dimension)
    ccp_strong = density_inside_quarter_radius / density_max_radius
    ccp_flag_strong = ccp_strong >= 1.2 or ccp_strong < 1
    return ccp_strong, ccp_flag_strong


def division_central_consistency(gb_list, gb_list_not):
    '''
    中心一致性划分
    Args:
        gb_list:
        gb_list_not:

    Returns:

    '''
    gb_list_new = []
    methods = '2-means'
    for gb in gb_list:
        if len(gb) > 1:
            ball_1, ball_2 = spilt_ball_k_means(gb, 2, methods)
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_not.append(gb)
                continue
            ccp, ccp_flag = get_ccp(gb)
            _, radius = calculate_center_radius(gb)
            sprase_parent = get_dm_sparse(gb)
            sprase_child1 = get_dm_sparse(ball_1)
            sprase_child2 = get_dm_sparse(ball_2)

            t1 = ccp_flag
            t4 = len(ball_2) > 2 and len(ball_1) > 2

            if (sprase_child1 >= sprase_parent or sprase_child2 >= sprase_parent) and (
                    len(ball_1) == 1 or len(ball_2) == 1):
                t4 = True
            if t1 and t4:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)

    return gb_list_new, gb_list_not
def calculate_center_radius(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDistances = np.sum(diffMat ** 2, axis=1)
    radius = max(sqDistances ** 0.5)
    return center, radius

def get_ccp(gb):
    '''
    得到中心一致性，中心一致性指平均半径内的样本密度与最大半径内的样本密度的比值，如果比值在1~1.3就不分裂，
    否则分裂为两个粒球。密度指：样本个数 / 半径 ** 维度。
    Args:
        gb:

    Returns:

    '''
    num = len(gb)
    if num == 0:
        return 0, False

    dimension = len(gb[0])
    center = gb.mean(axis=0)

    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5

    avg_radius = np.mean(distances)
    max_radius = np.max(distances)
    points_inside_avg_radius = np.sum(distances <= avg_radius)
    density_inside_avg_radius = points_inside_avg_radius / (avg_radius ** dimension)
    density_max_radius = num / (max_radius ** dimension)
    ccp = density_inside_avg_radius / density_max_radius
    ccp_flag = ccp >= 1.30 or ccp < 1  # 分裂
    return ccp, ccp_flag


def get_dm_sparse(gb):
    num = len(gb)
    dim = len(gb[0])

    if num == 0:
        return 0
    center = gb.mean(0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)

    sparsity = num / (radius ** dim)

    if num > 2:
        return sparsity
    else:
        return radius


def de_sparse(gb_list):
    avg_r_div_n = sum([get_radius(x) / len(x) for x in gb_list]) / len(gb_list)
    avg_r = sum([get_radius(x) for x in gb_list]) / len(gb_list)
    gb_list_new = []
    gb_split_list_new = []
    for gb in gb_list:

        r_t = get_radius(gb)
        if r_t / len(gb) > avg_r_div_n and r_t > avg_r:
            gb_split_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    for gb in gb_split_list_new:
        if len(gb) > 1:
            ball_1, ball_2 = spilt_ball(gb)
            gb_list_new.extend([ball_1, ball_2])
        else:
            gb_list_new.append(gb)
    return gb_list_new
def spilt_ball(data):
    ball1 = []
    ball2 = []
    A = pdist(data)
    d_mat = squareform(A)
    r, c = np.where(d_mat == np.max(d_mat))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if d_mat[j, r1] < d_mat[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])

    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]
def get_radius(gb):
    num = len(gb)
    center = gb.mean(0)
    diff_mat = center - gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius
def calculate_center_and_radius(gb):
    data_no_label = gb[:, :]  # 取坐标
    center = data_no_label.mean(axis=0)  # 压缩行，对列取均值  取出平均的 x,y
    data_no_label = data_no_label.numpy()
    center = center.numpy()
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))  # （x1-x1）**2 + (y1-y2)**2   所有点到中心的距离平均
    return center, radius

def get_ball_quality(gb, center):
    N = gb.shape[0]
    ball_quality =  N
    gb = gb.numpy()
    mean_r = np.mean(((gb - center) **2)**0.5)
    return ball_quality, mean_r


def ball_boundary_distance(centers, radii):
    # 计算球心之间的距离
    center_distances = pdist(centers)

    # 将球心之间的距离转换为方阵
    distance_matrix = squareform(center_distances)

    # 计算两球的半径和
    radius_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]

    # 计算边界距离（球心之间的距离减去半径和）
    boundary_distances = distance_matrix - radius_sum_matrix

    return boundary_distances


#     return ball_nearest_boundary
def ball_min_boundary_dist(ball_bound_distS, top_n):
    """
    计算每个粒球与其边界距离最近的 top_n 个粒球的索引。

    参数:
    ball_bound_distS: 每个粒球到其他粒球的边界距离矩阵
    top_n: 返回的最近粒球个数

    返回:
    返回一个列表，其中每个元素是一个包含对应粒球最近 top_n 个粒球索引的列表。
    """
    closest_balls = []

    # 遍历每个粒球的边界距离
    for i, distances in enumerate(ball_bound_distS):
        # 排除自己：将自身的距离设为一个很大的数，确保它不会被选中
        distances_copy = distances.copy()
        distances_copy[i] = float('inf')  # 将自身的距离设置为无穷大

        # 找出与其他粒球的距离并按升序排列，取前 top_n 个
        sorted_indices = np.argsort(distances_copy)[:top_n]

        closest_balls.append(sorted_indices)

    return closest_balls
def construct_adjacency_hat(adj):
    """
    :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray()
    return adj_normalized

def splits(gb_list, num, splitting_method):
    gb_list_new = []
    for gb in gb_list:
        p = get_num(gb)
        if p < num:
            gb_list_new.append(gb)#该粒球包含的点数小于等于num，那
        else:
            gb_list_new.extend(splits_ball(gb, splitting_method))#反之，进行划分，本来是[[1],[2],[3]]  变成[...,[1],[2],[3]]
    return gb_list_new

def splits_ball(gb, splitting_method):
    splits_k = 2
    ball_list = []

    # 数组去重
    len_no_label = np.unique(gb, axis=0)
    if splitting_method == '2-means':
        if len_no_label.shape[0] < splits_k:
            splits_k = len_no_label.shape[0]
        # n_init:用不同聚类中心初始化运行算法的次数
        #random_state，通过固定它的值，每次可以分割得到同样的训练集和测试集
        label = k_means(X=gb, n_clusters=splits_k, n_init=1, random_state=8)[1]  # 返回标签
    elif splitting_method == 'center_split':
        # 采用正、负类中心直接划分
        p_left = gb[gb[:, 0] == 1, 1:].mean(0)#求坐标平均值
        p_right = gb[gb[:, 0] == 0, 1:].mean(0)
        distances_to_p_left = distances(gb, p_left)#求出各点到平均点的距离
        distances_to_p_right = distances(gb, p_right)

        relative_distances = distances_to_p_left - distances_to_p_right
        label = np.array(list(map(lambda x: 0 if x <= 0 else 1, relative_distances)))

    elif splitting_method == 'center_means':
        # 采用正负类中心作为 2-means 的初始中心点
        p_left = gb[gb[:, 0] == 1, 1:].mean(0)
        p_right = gb[gb[:, 0] == 0, 1:].mean(0)
        centers = np.vstack([p_left, p_right])#[[],[]]
        label = k_means(X=gb, n_clusters=2, init=centers, n_init=10)[1]#以centers为中心进行聚类
    else:
        return gb
    for single_label in range(0, splits_k):
        ball_list.append(gb[label == single_label, :])#按照新打的标签分类
    return ball_list
def distances(data, p):
    return ((data - p) ** 2).sum(axis=1) ** 0.5

def get_num(gb):
    # 矩阵的行数
    num = gb.shape[0]
    return num
################################GB-DP#######################################
############################################################################

# features, gnd, p_labeled, p_unlabeled, adjs, adj_hats = dataloader(args.dataset_name, args.k, args.ratio)
features, gnd, p_labeled, p_unlabeled, adjs = dataprocess(args, args.dataset_name, args.k)

# print("ok")
llun = 1


features_sum = []
nfeat_sum =[]
adjtensor=[]
b = features.shape[1]
for jj in range(features.shape[1]):
    features_g = features[0][jj]
    features_g = torch.FloatTensor(features_g).float()
    nfeat_a = features_g.shape[1]
    # features_g = features_g.to(device)
    nfeat_sum.append(nfeat_a )
    features_sum.append(features_g)

#######################################GB-DP###########################
adj_nor = []
for dd in range(len(features_sum)):
    feature_to_index = {}
    index_counter = 0
    data = features_sum[dd]
    n_cluster = gnd[-1]
    gb_list_not_temp = []
    # gb_list = divide_ball_GBC_y(data, n_cluster, False)

    ##############################################################
    gb_list = [data]
    num = np.ceil(np.sqrt(data.shape[0]))
    while True:
        ball_number_1 = len(gb_list)  # 点数
        gb_list = splits(gb_list, num=num, splitting_method='2-means')
        ball_number_2 = len(gb_list)  # 被划分成了几个
        # gb_plot(gb_list)
        if ball_number_1 == ball_number_2:  # 没有划分出新的粒球
            break
############################################################################
    centers = []
    radiuss = []
    ball_num = []  # 粒球里面的元素个数
    ball_qualitys = []  # 每个粒球的质量
    mean_rs = []
    i = 0
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)  #nearest
        # center, radius = calculate_center_radius(gb)  #consistent
        centers.append(center)
        radiuss.append(radius)
        ball_num.append(gb.shape[0])
    centersA = np.array(centers)
    radiusA = np.array(radiuss)
    ball_numA = np.array(ball_num)
    # ball_qualitysA = np.array(ball_qualitys)  # 每一个粒球的半径和中心

    # 计算每个粒球的边界距离
    ball_bound_distS = ball_boundary_distance(centersA, radiusA)
    # 计算最小密度峰距离以及该点ball_min_dist  ball_min_distAD, ball_nearestAD
    # top_n = np.ceil(np.sqrt(data.shape[0])).astype(int)
    top_n =1
    ball_nearest = ball_min_boundary_dist(ball_bound_distS, top_n)

    features_all = np.vstack(gb_list)
    current_index = 0
    indices = []
#################################3边的连接#####################################
    for ball in gb_list:
        num_nodes = ball.shape[0]
        indices.extend(range(current_index, current_index + num_nodes))
        current_index += num_nodes
    indices = np.array(indices)

    # 4. 将索引列添加到特征矩阵的最后一列
    final_matrix = np.column_stack((features_all, indices))

    # 5. 初始化邻接矩阵
    num_total_points = data.shape[0]
    adj_matrix = np.zeros((num_total_points, num_total_points))

    # 6. 为每个粒球中的点添加边，并更新邻接矩阵
    for cc in range(len(gb_list)):
        # 获取当前粒子群体和最近的五个粒子群体
        nearest_indices = ball_nearest[cc]
        if len(nearest_indices) == 0:
            continue

        point1_gb = gb_list[cc]
        for idx in nearest_indices:
            point2_gb = gb_list[idx]  # 获取最近的粒子群体
            for point1 in point1_gb:
                for point2 in point2_gb:
                    point1_np = point1.numpy() if hasattr(point1, 'numpy') else point1
                    point2_np = point2.numpy() if hasattr(point2, 'numpy') else point2
                    # 获取与 point1 特征匹配的行索引
                    idx1 = np.where(np.all(final_matrix[:, :-1] == point1_np, axis=1))[0][0]  # 获取第一个匹配项

                    # 获取与 point2 特征匹配的第一个行索引
                    idx2 = np.where(np.all(final_matrix[:, :-1] == point2_np, axis=1))[0][0]  # 获取第一个匹配项
                    adj_matrix[idx1, idx2] = 1
                    adj_matrix[idx2, idx1] = 1

    # 7. 归一化邻接矩阵
    adj_matrix = construct_adjacency_hat(adj_matrix)
    adj_nor.append(adj_matrix)

features_sum1 = []
for ki in range(len(features_sum)):
    features_f = features_sum[ki]
    features_f = torch.FloatTensor(features_f).float()
    #features_f = features_f.to(device)
    features_sum1.append(features_f)

for j in range(len(adj_nor)):
    adja = adj_nor[j].astype(np.float32)
    adja = torch.tensor(adja, dtype=torch.float32)
    # adja = adj_nor[j].float()
    #adja = adja.to(device)
    adjtensor.append(adja)
#
gnd = torch.from_numpy(gnd).long()#.to(args.device)

model = MAUGCN(nfeat=nfeat_sum,
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=int(gnd.max()) + 1,
                dropout=args.dropout,
                phi=args.phi,
                lamda=args.lamda,
                alpha=args.alpha,
                variant=args.variant,
                )#.to(device)

optimizer = optim.Adam([
        {'params': model.params1, 'weight_decay': args.wd1},
        {'params': model.params2, 'weight_decay': args.wd2},
        # {'params': model.params3, 'weight_decay': args.wd3}
], lr=args.lr)


def train():

    model.train()
    optimizer.zero_grad()

    output,outputs,outputts  = model(features_sum1, adjtensor ,nfeat_sum)
    # losstrain3 = []
    # losstrain3_3 = 0
    # loss_train3 = []
    # loss_train3_3 = 0
    # for ii in range(len(outputts)):
    #     losstrain3.append(F.nll_loss(outputts[ii][p_labeled], gnd[p_labeled]))  # .to(device)
    #     losstrain3_3 += losstrain3[ii]
    #     loss_train3.append(F.mse_loss(outputts[ii][p_labeled], output[p_labeled]))
    #     loss_train3_3 += loss_train3[ii]
    # loss_train333 = loss_train3_3.cpu().detach().numpy()
    # loss_train3_sum = (1 - 0.2) * losstrain3_3 - 0.2 * loss_train333

    acc_train = accuracy(output[p_labeled], gnd[p_labeled])#.to(device))
    loss_train = F.nll_loss(output[p_labeled],gnd[p_labeled])#.to(device))
    loss_train.backward()
    # loss_train3_sum.backward()
    optimizer.step()

    return loss_train.item(), acc_train.item()


def test():
    model.eval()
    with torch.no_grad():
        output,outputs,outputts  = model(features_sum1, adjtensor,nfeat_sum)
        label_pre = []
        for idx in p_unlabeled:
            label_pre.append(torch.argmax(output[idx]).item())
        label_true = gnd[p_unlabeled].data.cpu()
        macro_f1 = f1_score(label_true, label_pre, average='macro')
        #loss_test = F.nll_loss(output[p_unlabeled], gnd[p_unlabeled].to(device))  # .to(device)
        #acc_test = accuracy(output[p_unlabeled], gnd[p_unlabeled].to(device))  # .to(device)
        loss_test = F.nll_loss(output[p_unlabeled], gnd[p_unlabeled])  # .to(device)
        acc_test = accuracy(output[p_unlabeled], gnd[p_unlabeled])  # .to(device)
        visual1(output, gnd, epoch, args.ratio,args.dataset_name)
        return loss_test.item(), acc_test.item(), macro_f1,output


# if args.test:
#     lose_test,acc_test =test()
acc_test_value = []
f1_test_value = []
loss_train_value = []
lamda_outputs = []

# 外部循环，遍历每个 lamda
output_values = []

# 内部循环，遍历 llun 次
for j in range(llun):
    accGCN = np.zeros((1, llun))
    timesGCN = np.zeros((1, llun))
    time = perf_counter()

    # 初始化最佳测试准确率和相关信息
    best_acc_test = 0
    best_epoch = 0
    best_f1 = 0

    # 训练过程，遍历每个 epoch
    for epoch in range(args.epochs):
        loss_train, acc_train = train()
        loss_test, acc_test, f1, output= test()

        # 存储每个 epoch 的训练和测试结果
        loss_train_value.append(loss_test)
        acc_test_value.append(acc_test)
        f1_test_value.append(f1)

        # 打印当前 epoch 的信息
        print(
            f"Epoch {epoch} - acc_train(GOC_GCN): {round(acc_train * 100, 1)}%, loss_train(GOC_GCN): {loss_train}, acc_test(GOC_GCN): {acc_test}, F1_test(GOC_GCN): {f1}")

        # 更新最佳的 acc_test 和相关信息
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            best_epoch = epoch
            best_f1 = f1

    loss_train, acc_train = train()
    loss_test, acc_test, f1, output= test()
    visual1(output, gnd, epoch, args.ratio, args.dataset_name)
    print(f"last- acc_train(GOC_GCN): {round(acc_train * 100, 1)}%, loss_train(GOC_GCN): {loss_train}, acc_test(GOC_GCN): {acc_test}, F1_test(GOC_GCN): {f1}")

    # 记录当前 lamda 对应的最佳结果
    output_values.append(( best_acc_test, best_f1))

# 将每个 lamda 对应的最佳结果加入到 lamda_outputs 列表中
lamda_outputs.extend(output_values)

# 打印每个 lamda 的最佳结果
for output, macro_f1 in lamda_outputs:
    print(f" Best acc_test: {round(output * 100, 1)}%, Best f1_score: {macro_f1}")

#########################loss###################
# smoothed_loss = savgol_filter(loss_train_value, window_length=11, polyorder=5)
# smoothed_test = savgol_filter(acc_test_value, window_length=11, polyorder=5)
# smoothed_f1 = savgol_filter(f1_test_value, window_length=11, polyorder=5)
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(range(args.epochs), smoothed_loss, label="Loss Test", color='red', linewidth=2)
# ax1.set_xlabel("The number of epochs",fontsize = '20')
# ax1.set_ylabel("Loss", color='black',fontsize = '20')
# ax1.tick_params(axis='y', labelcolor='black', labelsize = 20)
# ax1.tick_params(axis='x', labelcolor='black', labelsize = 20)
# ax1.set_ylim(0, 2.5)
# ax1.set_xlim(-10, 510)
# # plt.legend(loc='best', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize="10 ")
# ax2 = ax1.twinx()
# ax2.plot(range(args.epochs), smoothed_test, label="Acc Test", color='green', linewidth=2)
# ax2.plot(range(args.epochs), smoothed_f1, label="F1 Test", color='blue', linewidth=2)
# ax2.set_ylabel("Accuracy and F1-score", color='black',fontsize = '20')
# ax2.set_ylim(0, 1)  # 设置右侧纵轴范围为 0 到 1
# ax2.tick_params(axis='y', labelcolor='black', labelsize = 20)
# plt.grid(axis='y')
  # 显示上面的label
# 显示网格
# plt.grid(True)
# plt.legend(loc='best', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize="10 ")
# plt.savefig('./fig/loss/'+args.dataset_name+'_'+'.png', dpi=1000)
# # 显示图像
# plt.show()
print(args.dataset_name)

#############################################################################################333
################GB-DP####################3

