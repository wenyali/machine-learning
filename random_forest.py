
#!/usr/bin/env python
# _*_coding:utf-8 _*_


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/19"


pd.set_option("display.width",100)
file_url = "./doc/appointment.txt"
column_names = ["air_miles","play_game_percent","eat_ice_liter","type"]
data = pd.read_csv(file_url,sep="\t",header= None,names=column_names)


# 如果数据列中有需要将字符串转化为数字处理的，可以用到sklearn 包中的函数

le = LabelEncoder()
column_names.remove("eat_ice_liter")


# include_columns = ["paly_game_percent","type"]
# for name in column_names:
#     if name not in include_columns:
#         data[name] = le.fit_transform(data[name])


data = data[column_names]
x_scale = data.iloc[:,:-1]
y_scale = data["type"]
# print(y_scale)
x_train,x_test,y_train,y_test = train_test_split(x_scale,y_scale,test_size=0.3)
print(len(x_train))
print(len(x_test))

model = RandomForestClassifier(n_estimators=50,criterion="gini",max_depth=10)
model.fit(x_train,y_train)
y_pred = model.predict(x_train)
print("训练成功率：",sum(y_train == y_pred)/len(y_train))
y_pred = model.predict(x_test)
print("训练成功率：",sum(y_test == y_pred)/len(y_test))