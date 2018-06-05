#!/usr/bin/env python
# _*_coding:utf-8 _*_

# 本案例是商品的广告投放和销量的模型和数据挖掘
# 使用的技术点是线性回归

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
# preprocessing 中的 MinMaxScaler,Standar
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/31"

pd.set_option("display.max_columns",100)
pd.set_option("display.width",500)

file_url = "../doc/Advertising.csv"
data = pd.read_csv(file_url)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
model = LinearRegression()
model.fit(x,y)
y_pre = model.predict(x)


print("下面是各种衡量模型的标准：")
# 求得 绝对值平均值误差的三种方法
# error1 = np.mean(np.abs(y-y_pre))
# error2 = np.sum(np.abs(y-y_pre))/len(y)
error3 = mean_absolute_error(y,y_pre)
print("平均绝对值误差(MAE)：",error3)

# 求得 均方差 的三种方法
error1 = np.sum((y-y_pre)**2)/len(y)
error2 = np.mean((y-y_pre)**2)
error3 = mean_squared_error(y,y_pre)
print("均方差误差 (MSE)：",error3)


# 求得 均方差根率 方法
error1 = np.sqrt(mean_squared_error(y,y_pre))
print("均方差根率 (RMSE)：",error1)

# 求得平均错误率方法
error_rate = np.mean(np.abs(y-y_pre)/y)
print("平均错误率为：",error_rate)
print("\n上述模型的存在很大的改进问题，下面通过将数据分为测试集和训练集来获取模型相应标准\n")

# 将原数据集分为测试集和训练集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
model = LinearRegression()
model.fit(x_train,y_train)
# 通过预测返回的是一个数组类型的数据集,可以直接用索引来获取，但是 y_test 是一个 dataFrame 列，可以使用 iloc
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)
print("训练集（MAE):{}\n"
      "训练集 (MSE):{}\n"
      "训练集 (RMSE):{}\n"
      "训练集 (error_rate){}\n"
      "测试集 (MAE):{}\n"
      "测试集 (MSE):{}\n"
      "测试集 (RMSE):{}\n"
      "测试集 (error_rate){}\n".format(mean_absolute_error(y_train,y_train_predict),
                mean_squared_error(y_train,y_train_predict),
                np.sqrt(mean_squared_error(y_train,y_train_predict)),
                np.mean(np.abs(y_train,y_train_predict)/len(y_train)),
                mean_absolute_error(y_test,y_test_predict),
                mean_squared_error(y_test,y_test_predict),
                np.sqrt(mean_squared_error(y_test,y_test_predict)),
                np.mean(np.abs(y_test-y_test_predict)/len(y_test))
                )
      )

print("\n 下面进一步的对模型进行（随意）优化，这里准备对每个特征值进行项次的改变：\n")

# 让出现分母为 0 的数据修改一下数据
data[(data['Radio'] ==0) | (data['TV'] == 0)] = 0.0001

data['TV2'] = data['TV']**2
data['TV3'] = data['TV']**3
data['TV_LOG'] =np.log(data['TV'])
data['TV_REV'] = 1/data['TV']

data['Radio2'] = data['Radio'] ** 2
data['Radio3'] = data['Radio'] ** 3
data['Radio_LOG'] = np.log(data['Radio'])
data['Radio_REV'] = 1 / data['Radio']

x_cols_names = ['TV', 'Radio', 'TV2', 'TV3', 'TV_LOG', 'TV_REV', 'Radio2', 'Radio3', 'Radio_LOG', 'Radio_REV']
x = data[x_cols_names]
y = data['Sales']
# 我们根据数据可以发现数据集目前每个特征值对应的数据相差很大，并且特征值之间也有了相关系数
# print(x.corr())
# 对x 的数据集进行区间缩放，方法有很多，以此为例
mms = Normalizer()
x = mms.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
model = LinearRegression()
model.fit(x_train,y_train)
print("优化后的每个特征值所占有的系数：")
# print(model.coef_,model.intercept_)
for col_name,coef in zip(x_cols_names,model.coef_):
    print(col_name,"\t",coef)

y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)
plt.plot(x_test,y_test,"ro")
plt.plot(x_test,y_test_predict,"go")
plt.show()
print("训练集（MAE):{}\n"
      "训练集 (MSE):{}\n"
      "训练集 (RMSE):{}\n"
      "训练集 (error_rate){}\n"
      "测试集 (MAE):{}\n"
      "测试集 (MSE):{}\n"
      "测试集 (RMSE):{}\n"
      "测试集 (error_rate){}\n".format(mean_absolute_error(y_train,y_train_predict),
                mean_squared_error(y_train,y_train_predict),
                np.sqrt(mean_squared_error(y_train,y_train_predict)),
                np.mean(np.abs(y_train,y_train_predict)/len(y_train)),
                mean_absolute_error(y_test,y_test_predict),
                mean_squared_error(y_test,y_test_predict),
                np.sqrt(mean_squared_error(y_test,y_test_predict)),
                np.mean(np.abs(y_test-y_test_predict)/len(y_test))
                )
      )

