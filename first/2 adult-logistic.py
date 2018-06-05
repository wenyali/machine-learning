#!/usr/bin/env python
# _*_coding:utf-8 _*_

# 本文中被注解的代码都是特征提取和优化，这样能让模型泛化能力更好
# 使用 logistic 回归实现二分类问题


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
# 这个accuracy_score 是正确率，是用来作为模型好坏的评判之一
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

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



# 将列中的字符串转化为数字,这个步骤是可以省略的，这个是因为 LabelBinarizer 类能够将字符串类别和数字类别都转换成哑编码
# col_str_names = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
#              'race', 'sex', 'native-country']
#
# le = LabelEncoder()
# for index in col_str_names:
#     # 字符串到数字的映射
#     adult_data[index] = le.fit_transform(adult_data[index])

# 年龄分箱结果作为序号传入
bins = [0,16,24,30,40,55,65,100]
age_bin_result = pd.cut(adult_data['age'],bins=bins,labels=np.arange(len(bins)-1))
# print("pd.cut(adult_data['age']) = \n",age_bin_result)
adult_data['age'] = age_bin_result

# 数据中有些数据是不能进行线性运算的，只能代表类别，这类的数据成为 dummy
# 我们需要将这类编码设置为 one-hot 编码

# 需要做 one_hot 编码的列名称
col_onehot_names = ['age', 'workclass', 'education',  'marital-status', 'occupation',
                    'relationship','race', 'sex','native-country',]


# 不需要做 ont-hot 编码的列名称
col_not_onehot_names = ['education-num','capital-gain', 'capital-loss','hours-per-week']


# lb = LabelBinarizer()
# 让one-hot 编码和原数据联系在一起
# result = adult_data[col_not_onehot_names]
# for index in col_onehot_names:
#     one_hot = lb.fit_transform(adult_data[index])
#     t = pd.DataFrame(data=one_hot,columns=np.arange(one_hot.shape[1]))
#     result = pd.concat((result,t),axis=1)



# 另外还有一种可以直接使用pandas 库中的get_dummies() 进行onehot 编码, 这样的结果和上面直接使用 LabelBinarizer 库是一样的
result = pd.get_dummies(adult_data[col_onehot_names])
result =pd.concat((adult_data[col_not_onehot_names],result),axis=1)
# print(result[:2])
#下面这两个x,y 只能选择一个
x,y = result,adult_data["income"]
#x,y = adult_data[col_names[:-1]],adult_data[col_names[-1]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)

# 这里的两个参数，第一个 penalty :惩罚规则，默认是 L2 (该模型满足高斯分布)，C 是正则化系数λ的倒数,越小说明正则化越强。
# 这里是的参数是为了防止过拟合，而设置的。
model = LogisticRegression(penalty='l2',C=1.0)
model.fit(x_train,y_train)
y_train_pre = model.predict(x_train)
y_test_pre = model.predict(x_test)

print("训练集正确率：",accuracy_score(y_train,y_train_pre))
print("测试集正确率：",accuracy_score(y_test,y_test_pre))