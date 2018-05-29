#!/usr/bin/env python
# _*_coding:utf-8 _*_
from math import log

__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/15"
"""
本项目：判定鱼类和非鱼类
项目概述
根据以下 2 个特征，将动物分成两类：鱼类和非鱼类。

特征：

不浮出水面是否可以生存
是否有脚蹼
"""
def create_data():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def get_comentropy(data):
    number = len(data)
    labels_count={}
    for item in data:
        type = item[-1]
        labels_count[type] = labels_count.get(type,0)+1
    # print(labels_count)

    # 对于 label 标签的占比，求出label 标签的信息熵
    shannon_ent = 0.0
    for key in labels_count:
        # 使用所有标签的发生概率计算类别出现的概率
        prob = float(labels_count[key])/number
        # 计算信息熵
        shannon_ent -=prob*log(prob,2)
    return shannon_ent

def split_data(data,index,value):
    """就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中

    :param data: 待划分的数据集
    :param index: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return: index 列为value 的数据集【该数据集需要排除index 列】
    """
    ret_data = []
    for child in data:
        # 判定index 列的数值与给定的 value 相同
        if child[index] == value:
            # 新组成一个不包含 index列的其余数组
            reduced_child = child[:index]
            reduced_child = reduced_child.extend(child[index+1:])
            ret_data.append(reduced_child)

    return ret_data

def choose_best_feature_split(data):
    """选择最好的特征

    :param data: 数据集
    :return: 最优的特征列
    """
    # 求第一行有多少列的 feature ,最后一行是label 列
    num_features = len(data[0])-1
    # 数据集的原始信息熵
    base_entropy = get_comentropy(data)
    # 最优的信息增益值，和最优的 feature 编号
    best_info_gain,best_feature = 0.0,-1
    # 遍历所有的 features
    for i in range(num_features):
        # 获取对应的 feature 下的所有数据
        feature_list = [example[i] for example in data]
        unique_feature_list = set(feature_list)
        new_entroy = 0.0
        for value in feature_list:
            sub_data = split_data(data,i,value)
            # 计算概率
            prob = len(sub_data)/float(len(data))
            # 计算信息熵
            new_entroy +=prob*get_comentropy(sub_data)
        info_gain = base_entropy - new_entroy
        print('info_gain={},bestFeature={}'.format(info_gain,i))
        if info_gain>best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature






if __name__ == '__main__':
    data,labels = create_data()
    comentropy = get_comentropy(data)
    best_feature = choose_best_feature_split(data)
    print("最好的特征决策：{}".format(best_feature))

