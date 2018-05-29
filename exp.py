#!/usr/bin/env python
# _*_coding:utf-8 _*_
from sklearn import datasets
import numpy as np


__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/13"

#diabetes = datasets.load_diabetes()
# print(diabetes.data[:3,2])

# data[:,np.newaxis,2] 是取出二维数组中所有的第3列的属性，并返回一个二维的数组
#diabetes_X = diabetes.data[:,np.newaxis,2]
# print(diabetes_X[:3])
#data = np.ones(20).reshape(4,5)
# print(data)
# print(data[:,1:3])

#a = data[:,np.newaxis,1:3]
# print(a)
# print(a.shape)

# a  = zip(('a','b','c','d'),(1,2,3,4,5))
# for item in a:
#     print(item)
#
# for item in a:
#     print(item)

# 请百度查询一下： extend和append的区别
# music_media.append(object)
# 向列表中添加一个对象object
# music_media.extend(sequence)
# 把一个序列seq的内容添加到列表中(跟 += 在list运用类似， music_media += sequence)
# 1、使用append的时候，是将object看作一个对象，整体打包添加到music_media对象中。
# 2、使用extend的时候，是将sequence看作一个序列，将这个序列和music_media序列合并，并放在其后面。
# music_media = []
# music_media.extend([1, 2, 3])
# print
# music_media
# 结果：
# [1, 2, 3]

# music_media.append([4, 5, 6])
# print
# music_media
# # 结果：
# # [1, 2, 3, [4, 5, 6]]
#
# music_media.extend([7, 8, 9])
# print
# music_media
# # 结果：
# # [1, 2, 3, [4, 5, 6], 7, 8, 9]
# '''

# a = np.array([[1],[2]])
# b = np.array([[11],[22]])
#
#
# print(a)
# print(b)
# # 纵合并
# c = np.hstack((a,b))
#
# # 横合并
# d = np.vstack((a,b))
# print(c)
# print(d)
# print(c[0][1])

from sklearn import linear_model
reg = linear_model.LinearRegression()
model = reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(model)
print(reg.coef_)
