#!/usr/bin/env python
# _*_coding:utf-8 _*_
from random import random

from numpy import mat
import numpy as np

__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/22"


# 从 fiel_url 中加载数据
def load_data_set(file_url):
    data_mat = []
    fr = open(file_url)
    for line in fr.readlines():
        cur_cine = line.strip().split('\t')
        flt_line = map(float, cur_cine)
        data_mat.append(flt_line)
    return data_mat


# 计算两个向量的欧式距离
def dist_eclud(veca,vecb):
    return np.sqrt(sum(pow(veca-vecb,2)))


# 构建一个包含K 个随机质心的集合
def rand_cent(data_set,k):
    n = data_set.shape[1]
    centroids = mat(np.zeros((k,n)))
    # 下面这个for 循环就是得出每个 k 中每一个特征上挑选一个随机合适的特征值
    for j in range(n):
        min_j = min(data_set[:,j])
        range_j = float(max(data_set[:,j])-min_j)
        # 这里是 每个 k 个j 位置上赋值为一个随机范围的特征值
        centroids[:,j] = mat(min_j+range_j*random.rand(k,1))
    return centroids

def k_means(data_set,k,dist_means = dist_eclud,create_cent = rand_cent):
    m = data_set.shape[0]
    cluster_assment = mat(np.zeros((m,2)))
    centeroids = create_cent(data_set,k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dis_ji =dist_means(centeroids[j,:],centeroids[i,:])
                if dis_ji < min_dist:
                    min_dist = dis_ji
                    min_index = j
            if cluster_assment[i,0] != min_index:
                cluster_changed = True
                cluster_assment[i,:] = min_index,min_dist**2

        for cent in range(k):
            pts_in_clust = data_set[np.nonzero(cluster_assment[:,0].A == cent)[0]]
            centeroids[cent,:] = np.mean(pts_in_clust,axis=0)
    return centeroids,cluster_assment



if __name__ == '__main__':
    file_url = "../python_machine_learning/无监督学习/testSet.txt"
    data_mat = mat(load_data_set(file_url))
    print(data_mat.shape)
    #rand_cent(data_mat,3)

    # 这里是mat 和map 的组合测试
    # flt_line = map(float, [1,2])
    # a = mat([[1,2],[1,2],[1,2],[1,2]])
    # b = mat([flt_line,flt_line,flt_line,flt_line])
    # print(a.shape,b.shape)