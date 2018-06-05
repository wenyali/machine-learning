#!/usr/bin/env python
# _*_coding:utf-8 _*_

# 本文中被注解的代码都是特征提取和优化，这样能让模型泛化能力更好
# 使用 logistic 回归实现二分类问题
from time import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/31"

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',500)


adult_url = "../doc/adult/adult.data"

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']


adult_data = pd.read_csv(adult_url,header=None,names=col_names)

#"fnlwgt" 这个字段是不需要的
adult_data.drop("fnlwgt",axis=1,inplace=True)
col_names.remove("fnlwgt")


# 年龄分箱结果作为序号传入
bins = [0,16,24,30,40,55,65,100]
age_bin_result = pd.cut(adult_data['age'],bins=bins,labels=np.arange(len(bins)-1))
adult_data['age'] = age_bin_result

# 数据中有些数据是不能进行线性运算的，只能代表类别，这类的数据成为 dummy,我们需要将这类编码设置为 one-hot 编码

# 需要做 one_hot 编码的列名称
col_onehot_names = ['age', 'workclass', 'education',  'marital-status', 'occupation',
                    'relationship','race', 'sex','native-country',]


# 不需要做 ont-hot 编码的列名称
col_not_onehot_names = ['education-num','capital-gain', 'capital-loss','hours-per-week']

# 直接使用pandas 库中的get_dummies() 进行onehot 编码
result = pd.get_dummies(adult_data[col_onehot_names])
result =pd.concat((adult_data[col_not_onehot_names],result),axis=1)

x,y = result,adult_data["income"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)



# 这里使用的模型是 logistic回归 这里有两个常用的参数：
# 第一个 penalty :惩罚规则，默认是 L2 (该模型满足高斯分布)，
# 第二个 C 是正则化系数λ的倒数,越小说明正则化越强。
model = LogisticRegression(penalty='l2',C=1.0)
model.fit(x_train,y_train)
y_train_pre = model.predict(x_train)
y_test_pre = model.predict(x_test)
print("使用 logistic 回归模型得到的结果：")
print("训练集正确率：",accuracy_score(y_train,y_train_pre))
print("测试集正确率：",accuracy_score(y_test,y_test_pre))


# adult-logistic.py 文件中，我们使用的是 logistic 回归解决的分类问题，在这里使用的是 决策树模型
# 首先，使用 GridSerchCV 库对模型进行交叉验证，得到一个最优的参数

model = DecisionTreeClassifier(criterion='gini',max_depth=13)
model.fit(x_train,y_train)
y_train_pre = model.predict(x_train)
y_test_pre = model.predict(x_test)
print("\n使用决策树模型没有调参得到的结果：")
print("训练集正确率：",accuracy_score(y_train,y_train_pre))
print("测试集正确率：",accuracy_score(y_test,y_test_pre))




# 利用交叉验证，得到最优的参数
# model = DecisionTreeClassifier(criterion='gini',max_depth=13,min_samples_leaf=5,min_samples_split=7,min_impurity_decrease=0.001)
# model = GridSearchCV(model,param_grid={"max_depth":np.arange(7,14),"min_samples_split":np.arange(3,10),
#                                        'min_samples_leaf':np.arange(3,8)},cv=3)
# t0 = time()
# model.fit(x_train,y_train)
# t1 = time()
# t = t1-t0
# print("\n训练耗时：{}秒".format(t))
# print(model.best_params_)


model = DecisionTreeClassifier(criterion='gini',max_depth=7,min_samples_split=3,min_samples_leaf=3,min_impurity_decrease=0.01)
model.fit(x_train,y_train)
y_train_pre = model.predict(x_train)
y_test_pre = model.predict(x_test)
print("\n使用决策树模型调参得到的结果：")
print("训练集正确率：",accuracy_score(y_train,y_train_pre))
print("测试集正确率：",accuracy_score(y_test,y_test_pre))


print("\n利用决策树模型来说，不需要修改参数的数值，修改之后反而又不好了，也从另方面说，我们需要优化各种参数")