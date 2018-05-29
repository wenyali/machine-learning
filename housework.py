#!/usr/bin/env python
# _*_coding:utf-8 _*_

import numpy as np

__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/15"

np.random.seed(0)
data = np.random.rand(40,1)
# 将所有的数据进行排列
X = np.sort(5*data,axis=0)
y = np.sin(X).ravel()
print(y)