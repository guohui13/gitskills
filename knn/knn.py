# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     knn
   Description :
   Author :       gh
   date：         2019-07-09    
-------------------------------------------------
   Change Activity:
                   19-7-9:
-------------------------------------------------
"""
__author__ = 'gh'

from numpy import *
import operator


# 创建一个数据集和标注
def createDataSet():
    try:
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']

    except Exception as e:
        print(e)
    return group, labels


# 简单分类器
def classify0(inX: list, dataSet: ndarray, labels: list, K: int) -> str:
    dataSetSize = dataSet.shape[0]  # 矩阵长度 数据集的长度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 构建一个ndarray ，重复第二个参数的形状格式
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # argsort 按照ndarray的索引位置升续排序
    classCount = {}  # 字典记录标签和类别出现的次数
    for i in range(K):
        voteIlabel = labels[sortedDistIndicies[i]]  # k近邻的k个排前的
        classCount[voteIlabel] = classCount.setdefault(voteIlabel, 0) + 1  # 记录标签出现的次数
    # 按照标签排序次数逆排序 ↓
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    groups, labels = createDataSet()
    k = classify0([1, 1], groups, labels, 3)
    print("iput belongs class {}".format(k))
