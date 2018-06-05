#!/usr/bin/env python
# _*_coding:utf-8 _*_
from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier






# 如果读取的文档没有头部信息，我们尽量的去用numpy 来读取，在大量的数据存在的时候，我们需要考虑并发的问题，
# pandas 的结构比 numpy 的复杂，运行时占用的 cpu 比较高。

if __name__ == "__main__":
    print("Load Training File Start..............................")
    file_url = "../doc/handwritng/optdigits.tra"
    data_train =  np.loadtxt(file_url,dtype=np.float,delimiter=',')
    # 根据第二个参数将数据数据分为几部分
    x_train,y_train = np.split(data_train,(-1,),axis=1)
    # -1 是代表在 8*8 合并组之后，整体组成的最大组数
    train_images = x_train.reshape(-1,8,8)
    print('train_images.shape = ', train_images.shape)
    # (y.ravel().astype(np.int)  等同于 y.reshape(-1).astype(np.int)  将多维数据变化为一维数组
    y_train = y_train.ravel().astype(np.int)


    print("Load Testing File Start..............................")
    file_url = "../doc/handwritng/optdigits.tes"
    data_test = np.loadtxt(file_url,dtype=np.float,delimiter=',')
    x,y = np.split(data_test,(-1,),axis=1)

    # 进行交叉验证和测试集的划分，这个划分都是在测试集中划分
    x_cv,x_test, y_cv, y_test = train_test_split(x,y,test_size=0.8)
    x_cv_image = x_cv.reshape(-1,8,8)
    x_test_image = x_test.reshape(-1,8,8)
    y_cv =y_cv.reshape(-1).astype(np.int)
    y_test = y_test.reshape(-1).astype(np.int)

    print('cv_images.shape = ', x_cv_image.shape)
    print('test_images.shape = ', x_test_image.shape)
    print('Load Data OK...')
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 训练数据的展示
    # figsize 是定义整个图片的大小，
    # plt.figure(figsize=(20,9))
    # for index,image in enumerate(train_images[:16]):
    #     plt.subplot(2,8,index+1)
    #     plt.imshow(image,cmap='gray',interpolation="nearest")
    #     plt.title("训练第{}张手写体,数字为{}".format(index+1,y_train[index]))
    # plt.show()
    #
    # # 测试数据的展示
    # plt.figure(figsize=(20,9))
    # for index,image in enumerate(x_test_image[:16]):
    #     plt.subplot(2,8,index+1)
    #     plt.imshow(image,cmap='gray',interpolation='nearest')
    #     plt.title("测试第{}张手写体,数字为{}".format(index+1,y_test[index]))
    # plt.tight_layout()  # 这个只是显示子图片上下具体的问题，属于视觉上的代码
    # plt.show()



    # 这里是使用 svm 进行手写字体的识别的，交叉正确率是98%，测试集正确率是98%
    print("这里是使用 svm 模型 进行手写字体的识别")
    # params = {'C': np.logspace(0,3,7), 'gamma':np.logspace(-5,0,11)}
    # # 这里，我们是利用一个 3次交叉验证来暴力得出一个最优参数解，最终的结果是 C=31.628
    # model = svm.SVC()
    # model = GridSearchCV(model,param_grid=params,cv=3)
    # t0 = time()
    # model.fit(x_train,y_train)
    # t1 = time()
    # t = t1-t0
    # print("训练耗时：{}秒".format(t))
    # print(model.best_params_)

    model = svm.SVC(C=31.6227, gamma=0.00046)
    model.fit(x_train,y_train)
    y_train_pre = model.predict(x_train)
    print("训练集的测试准确率：{}".format(accuracy_score(y_train, y_train_pre)))

    y_cv_pre = model.predict(x_cv)
    print("交叉集的测试准确率：{}".format(accuracy_score(y_cv, y_cv_pre)))

    y_test_pre = model.predict(x_test)
    print("测试集的测试准确率：{}".format(accuracy_score(y_test, y_test_pre)))

    # x_cv_error_image= x_cv_image[y_cv_pre != y_cv]
    # y_cv_error = y_cv[y_cv_pre != y_cv]
    # y_cv_pre_error = y_cv_pre[y_cv_pre != y_cv]
    # print(len(y_cv_error),len(y_cv_pre_error))
    #
    # plt.figure(figsize=(10,8),facecolor='w')
    # for index,image in enumerate(x_cv_error_image):
    #     plt.subplot(4,3,index+1)
    #     plt.imshow(image,cmap='gray',interpolation='nearest')
    #     plt.title("第{}张，数字为{}，测试数字为{}".format(index+1,y_cv_error[index],y_cv_pre_error[index]))
    # plt.suptitle(" 手写体错误展示")
    # plt.show()




    print("\n下面是使用决策树进行手写字体的识别分类问题")
    model = RandomForestClassifier(n_estimators=102,max_depth=12)
    #model = GridSearchCV(model,param_grid={"n_estimators":np.arange(95,105),"max_depth":np.arange(12,17)},cv=3)
    # t0 = time()
    # model.fit(x_train,y_train)
    # t1 = time()
    # t =t1-t0
    # print("训练耗时：{}秒".format(t))
    # print(model.best_params_)

    model.fit(x_train, y_train)
    y_train_pre = model.predict(x_train)
    print("训练集的测试准确率：{}".format(accuracy_score(y_train, y_train_pre)))

    y_cv_pre = model.predict(x_cv)
    print("交叉集的测试准确率：{}".format(accuracy_score(y_cv, y_cv_pre)))

    y_test_pre = model.predict(x_test)
    print("测试集的测试准确率：{}".format(accuracy_score(y_test, y_test_pre)))


