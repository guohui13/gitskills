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

# 创建一个数据集和标注
def createDataSet():
    try:
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']

    except Exception as e:
        print(e)
    return group, labels

def classify0(inX:list, dataSet:ndarray, labels:list, K:int)->str:
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #构建一个ndarray ，重复第二个参数的形状格式
    sqDiffMat = diffMat**2
    distances = sqDiffMat**0.5
    sortedDistIndicies = distances.argsort() # argsort 按照ndarray的索引位置排序
    classCount={}
    for i in range(K):



