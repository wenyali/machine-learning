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

x = np.linspace(-50,50,num=100)
y = x**2+20*x
# plot 第三个参数共有三种位置：
# 第一个位置：表示使用的 color  [g,r,b,....]
# 第二个位置:表示绘制曲线的组成  [-,o,+] 三种模式  默认是[-] 线性
# 第三个位置：表示虚线或者实线  dashed
# 完整的表达式为：
#  plot(x, y, color='green', marker='o', linestyle='dashed',
#                 linewidth=2, markersize=12)

# plt.plot(x,y,"g-",linewidth=3)
# plt.show()
#
# x=-5
# a=0.01
# for i in range(1000):
#     x -= a*(2*x+20)
#     print(i,x)






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

#
# pd.set_option('display.width',400)
# x = np.random.random((200,5))
# x = pd.DataFrame(x)
# x[50] = x[0]*3
# x[60] = x[1]*(-2)
# x[70] = x[3] +np.random.randn(200)*0.01
# x[80] = x[0]+x[1]
#
#
# print(x.corr())
# a = np.random.rand(2,3,4,5)
# print(type(a),a)

# 这里的 -1 代表的含义就是维度的最大值（6），类似于索引一样
# a = np.arange(0,60,10).reshape(-1,1)+np.arange(6)
# print(a)

# a = np.sum(1/(np.arange(1,101,1)**2))
# print(a)

# from sklearn.preprocessing import StandardScaler
# import numpy as np
# x = np.linspace(1,10,num=10).reshape(5,2)
# x_standared = StandardScaler().fit_transform(x)
# print("原数据为：\n",x,"\n","标准化后的数据为：\n",x_standared)

# from sklearn.preprocessing import MinMaxScaler
# import  numpy as np
# x = np.linspace(1,10,num=10).reshape(5,2)
# x_standared = MinMaxScaler().fit_transform(x)
# print("原数据为：\n",x,"\n","标准化后的数据为：\n",x_standared)

# from sklearn.preprocessing import Normalizer
# import  numpy as np
# x = np.linspace(1,10,num=10).reshape(5,2)
# x_standared = Normalizer().fit_transform(x)
# print("原数据为：\n",x,"\n","归一化后的数据为：\n",x_standared)

# from sklearn.preprocessing import Binarizer
# import  numpy as np
# x = np.linspace(1,10,num=10).reshape(5,2)
# x_standared = Binarizer(5).fit_transform(x)
# print("原数据为：\n",x,"\n","二值化后的数据为：\n",x_standared)

# from sklearn.preprocessing import OneHotEncoder
# x = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
# x_standared = OneHotEncoder(sparse=False).fit_transform(x)
# print("原数据为：\n",x,"\n","哑编码后的数据为：\n",x_standared)
#
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import LabelBinarizer
#
# pd.set_option('display.width',300)
# pd.set_option('display.max_columns',300)
# testdata = pd.DataFrame({'pet': ['cat', 'dog', 'dog', 'fish'],
# 'age': [4 , 6, 3, 3],
# 'salary':[4, 5, 1, 1]})
# age_and_salary = OneHotEncoder(sparse= False).fit_transform(testdata[['age',"salary"]])
# age_and_salary = pd.DataFrame(age_and_salary,columns=['age_3','age_4','age_6','salary_1','salary_4','salary_5'])
# print(age_and_salary)
# # 对于含有字符串的特征，有下面两种方法解决：
# # 第一：LabelEncoder() + OneHotEncoder()
# pet = OneHotEncoder(sparse=False).fit_transform(LabelEncoder().fit_transform(testdata['pet']).reshape(-1,1))
# pet = pd.DataFrame(pet,columns=['pet1','pet2','pet3'])
# print(pd.concat((testdata,pet,age_and_salary),axis=1))
#
# ## 第二：直接调用 LabelBinarizer()，用于数字和字符串都是可以的
# pet_binarlizer = LabelBinarizer().fit_transform(testdata.pet)
# pet = pd.DataFrame(pet_binarlizer,columns=['pet_cat','pet_dog','pet_fish'])
# print(pet)
#
# # 除了上面的方法，这里还有一个最简单的方法，就是调用 pandas 库 中 get_dummies() 方法
# print(pd.get_dummies(testdata,columns=testdata.columns))


# 缺失值计算
# from sklearn.preprocessing import Imputer
# from numpy import vstack,array,nan,hstack,max
# data = vstack((array([nan,nan,None]),array([[1,2,3],[4,5,6]])))
# # 默认情况下是 mean  总共是有三种的选择 ['mean', 'median', 'most_frequent']
# data = Imputer(strategy='mean').fit_transform(data)
# print(data)

# 数据变化
# from sklearn.preprocessing import PolynomialFeatures
# # 多项式特征类 默认情况下对数据进行度为2 的转换
# data = PolynomialFeatures(degree=2).fit_transform(np.linspace(1,10,num=10).reshape(-1,1))
# data = pd.DataFrame(data,columns=['x0','x','x^2'])
# print(data)

# from numpy import log1p,log,log2,log10,poly
# from sklearn.preprocessing import FunctionTransformer
# x = np.linspace(1,10,num=10).reshape(-1,1)
# # log() 是 e 为底数的对数
# data_log = FunctionTransformer(log).fit_transform(x)
# data_log2 = FunctionTransformer(log2).fit_transform(x)
# data_log10 = FunctionTransformer(log10).fit_transform(x)
# data = pd.DataFrame(np.hstack((x,data_log,data_log2,data_log10)),columns=['x','loge','log2','log10'])
# print(data)