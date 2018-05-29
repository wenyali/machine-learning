#!/usr/bin/env python
# _*_coding:utf-8 _*_

from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/21"

iris = datasets.load_iris()

# iris 主要是包含了两部分的内容:
    # 一部分是 150 个样本的特征数据 iris.data
    # 一部分是 150 个样本的分类     iris.target
#print(len(iris.data),"\n",iris.data)
#print(len(iris.target),"\n",iris.target)


# 无量钢化

# 1、正态分布： 使用 preprocessing 库 中的 StandardScaler
# from sklearn.preprocessing import StandardScaler
# data = StandardScaler().fit_transform(iris.data)
# print(data)

# 2、区间缩放法：使用 preprocessing 库的 MinMaxScaler 类对数据进行区间缩放
from sklearn.preprocessing import MinMaxScaler
data = MinMaxScaler().fit_transform(iris.data)
print(data)

# 3、归一化:使用 preprocessing 库中的Normalizer 类对数据进行归一化的操作
# from sklearn.preprocessing import Normalizer
# # data = Normalizer().fit_transform(iris.data)
# # print(data)


