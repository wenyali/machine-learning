#!/usr/bin/env python
# _*_coding:utf-8 _*_

import  matplotlib.pyplot as plt
import numpy as np
from scipy.misc import comb
import pandas as pd

__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/29"

# x*x 的函数图像
# x = np.linspace(0,5,50)
# y = x**x
# plt.plot(x,y,"r","3")
# plt.show()



# 利用梯度下降求取 x**x 函数的最小值
# x =1
# alpha = 0.1
# for i in range(200):
#     # x**x*(np.log(x)+1)  这个就是x**x 的导数
#     # alpha*x**x*(np.log(x)+1)  这个是学习率乘以导数得出的梯度下降的步长
#     x -= alpha*x**x*(np.log(x)+1)
#     print(i,x)



# 求取函数 的最小值
# x =1
# alpha = 0.01
# for i in range(200):
#     # (-2*np.sin(2*x) + 6*(np.cos(3*x)))  这个就是 sin2x +2cos3x 的导数
#     # alpha*(-2*np.sin(2*x) + 6*(np.cos(3*x)))  这个是学习率乘以导数得出的梯度下降的步长
#     x -= alpha * (-2*np.sin(2*x) + 6*(np.cos(3*x)))
#     print(i,x)


# x =  np.random.random(100)
# plt.hist(x,20,color="g",edgecolor="k")
# plt.show()


# 下面这个函数就是一个 当x>0 时，y=x  当 x <0 ，y = 0
# x = np.linspace(-3,3,500)
# y = np.log(1+np.exp(x))
# plt.plot(x,y,"r-",linewiths=3)
# plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(111)
# u = np.linspace(0,4,1000)
# x,y = np.meshgrid(u,u)
# z = np.log(np.exp(x) + np.exp(y))
# ax.contourf(x,y,z,30)
# plt.show()


# 从n 个二分类器中选择 个，得出的最终结果
# def multi_clf_prob(n,p):
#     q = 0
#     for k in range(n //2 +1,n+1):
#         q+=comb()
#
# M = 121
# x = []
# p = []
# for n in range(1,M,2):
#     x.append()
#     p.append(multi_clf_prob(n,0.6))


# 暴力模拟赢球的概率，解释一个道理，当每次赢得概率是0.6 ，那么多次测试之后，我们赢的机会将会得到0.827
# p = 0.6
# m = 10000
# na = 0
# bn = 0
# for i in range(m):
#     a ,b = 0,0
#     while True:
        # 这是概率的模拟
#         if np.random.random() < p :
#             a +=1
#         else:
#             b+=1
#         if a == 11:
#             na+=1
#             break
#         elif b ==11:
#             bn+=1
#             break
# print("a 赢的次数：",na,"赢的概率是：",na/m)
# print("b 赢的次数：",bn,"赢的概率是：",bn/m)


# 计算本福特定律：首位出现[1-9] 的分别概率



# 计算均值，方差，协方差（for 循环）


pd.set_option('display.width',400)
x = np.random.random((200,5))
x = pd.DataFrame(x)
x[50] = x[0]*3
x[60] = x[1]*(-2)
x[70] = x[3] +np.random.randn(200)*0.01
x[80] = x[0]+x[1]


print(x.corr())