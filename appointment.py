#!/usr/bin/env python
# _*_coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import operator

__title__ = ''
__author__ = "wenyali"
__mtime__ = "2018/5/15"


def file2matrix(filename):

    """ 将文本文件中的数据转化为numpy 的矩阵形式

    :param filename: 所需要的约会样本的文本文件
    :return:numpy matrix
    """

    with open(filename,"rt") as f:
        number = f.readlines()
        number_lines = len(number)
        # 创建一个和文件行长度相同的维度且3列空矩阵
        mat = np.zeros((number_lines,3))
        class_label_vector = []
        for index,line in enumerate(number):
            # 去除两端空格并以tab 分割
            line = line.strip()
            number_list = line.split("\t")
            mat[index,:] = number_list[:3]
            class_label_vector.append(int(number_list[-1]))

    return mat,class_label_vector

def draw_picture(matrix,class_label_vector):

    """根据 matrix 数据绘制散点图

    :param matrix: 绘制图需要的数据
    :param class_label_vector:数据的分类，也是散点图中用的颜色分类和点的大小
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 将原有的数据扩大15倍，这样得到的点比较大，容易看清
    size = 15.0*np.array(class_label_vector)
    # 这里是将第1列数据和第2列数据绘制散点图，可以很清晰的看出三类的区别
    ax.scatter(matrix[:,0],matrix[:,1],s=size,c=size)

def normal_data(matrix):

    """归一化特征值，消除特征值之间量级不同导致的影响

    :param matrix:数据集
    :return:归一化之后的数据集 normal_matrix,ranges(范围) 和 min_vals（最小值）
    """
    min_vals = matrix.min(axis=0)
    max_vals = matrix.max(axis=0)

    # 计算差值
    cha = max_vals - min_vals

    # 这里本身是不需要的，因为 numpy 在运算的时候，会进行广播式的运算
    # m = matrix.shape[0]
    # m_mat = np.tile(min_vals,(m,1))

    normal_matrix = (matrix-min_vals)/cha

    return normal_matrix,cha,min_vals





# KNN 算法的伪代码
    # 对于每一个在数据集中的数据点：
    # 计算目标的数据点与该数据点的距离
    # 将距离从小到大排列
    # 选取前 K 个最短的距离
    # 选取这个k 个中最多的分类类别
    # 返回该类别作为目标数据点的预测值

def classify(inx,matrix,labels,k):

    # 距离的获取，这里使用的是欧式距离
    sq_distance = (inx - matrix)**2
    distance = sq_distance.sum(axis=1)**0.5

    # 将距离进行从小到大的排序，这里返回的是从小到大，原数据的索引
    sorted_distance = distance.argsort()

    # 选取最前k个距离，并选取这 K 个类别最多的分类类别
    class_count={}
    for i in range(k):
        # labels中每个类别元素和距离中每个元素是一一对应关系
        label = labels[sorted_distance[i]]
        # 将class_count 存在的label键值取出来(默认0)，然后加1
        class_count[label] = class_count.get(label,0)+1

    # sorted 接受三个参数，第一个是迭代对象，给出的实参是class_count.items(),第二个参数获取比较的元素，第三个是排序的顺序
    # 这个sorted 返回值是一个含有 tuple 元素的数组
    sorted_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    # 返回排在第一位，也就是出现类别最多的分类
    return sorted_class_count[0][0]

def appointment_test():
    """
    对约会模型的测试
    :return: 测试数据在模型中出现的错误数
    """

    # 设置一个测试数据的比列
    test_rate = 0.1

    filename = "./doc/appointment.txt"
    matrix, class_label_vector = file2matrix(filename)
    normal_matrix, cha, min_vals = normal_data(matrix)
    m = normal_matrix.shape[0]
    test_num = int(m*test_rate)
    # print("测试样本的数量：",test_num)

    error_count = 0.0
    for i in range(test_num):
        label = classify(normal_matrix[i,:],normal_matrix[test_num:m,:],class_label_vector[test_num:m],3)
        #print ("the label test: %d, real label is: %d" % (label, class_label_vector[i]))
        if label != class_label_vector[i]:
            error_count+=1

    return (error_count / float(test_num))

# 做测试
def main():
    resultList = {1:'not at all', 2:'in small doses', 3:'in large doses'}
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    filename = "./doc/appointment.txt"
    matrix, class_label_vector = file2matrix(filename)
    normal_matrix, cha, min_vals = normal_data(matrix)

    test_demo = np.array([ffMiles, percentTats, iceCream])
    test_demo = (test_demo-min_vals)/cha
    label = classify(test_demo,normal_matrix,class_label_vector,3)
    print("You will probably like this person: ", resultList.get(label))


if __name__ == '__main__':

    # 约会模型做测试
    error_rate = appointment_test()
    print("本模型的测试误差有：",str(error_rate*100),"%")

    # 现实情况做测试
    print("""
    约会类型分为三种：
    1、not at all
    2、in small doses
    3、in large doses
    """)
    print("现在测试你属于哪一种？输入值为0~1")
    main()


