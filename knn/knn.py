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

import operator

import matplotlib.pyplot as plt
from numpy import *


# 创建一个数据集和标注
def createDataSet()-> 'ndarray,list':
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
    sqDistances = sqDiffMat.sum(axis=1) #朝着变换的下标来
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # argsort 按照ndarray的索引位置升续排序
    classCount = {}  # 字典记录标签和类别出现的次数
    for i in range(K):
        voteIlabel = labels[sortedDistIndicies[i]]  # k近邻的k个排前的
        classCount[voteIlabel] = classCount.setdefault(voteIlabel, 0) + 1  # 记录标签出现的次数
    # 按照标签排序次数逆排序 ↓
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]





# 文件转换矩阵，返回分类数据和数据标签
def file2matrix(filename:str)->ndarray:
    dic={'largeDoses':3, 'smallDoses':2, 'didntLike':1} #分类标签的字符和数字的同一描述
    try:
        with open(filename) as fr:
            arrayOlines = fr.readlines() #文件缓存
            numberOfLines = len(arrayOlines) #文件长度
            returnMat = zeros((numberOfLines,3)) #构造0矩阵
            classLabelVector = []
            index = 0

            for line in arrayOlines:
                line = line.strip()
                listFromLine = line.split('\t')
                returnMat[index,:] = listFromLine[0:3]
                if listFromLine[-1].isdigit():
                    classLabelVector.append(int(listFromLine[-1]))
                else:
                    classLabelVector.append(int(dic.get(listFromLine[-1])))
                index += 1
            return returnMat,classLabelVector
    except Exception as e:
        print(e)


def analysis_data():
    pass



if __name__ == '__main__':
    groups, labels = createDataSet()
    k = classify0([1, 1], groups, labels, 3)
    print("iput belongs class {}".format(k))
    datingDataMat,datingLabels = file2matrix(r'./data/datingTestSet.txt')
    print(datingDataMat)
    print("----------------------datingLabels---------------------")
    print(datingLabels)
    fig1=plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(datingDataMat[:,1],datingDataMat[:,2])
    plt.show()
    fig2=plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0*array(datingLabels),
               15.0*array(datingLabels))
    plt.show()