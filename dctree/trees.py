# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     trees
   Description :
   Author :       gh
   date：         2019-07-16    
-------------------------------------------------
   Change Activity:
                   19-7-16:
-------------------------------------------------
"""
__author__ = 'gh'


from numpy import *
import operator


def createdataset()-> list:
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset,labels

def calcShannonEnt(dataset) -> float :  # 计算熵的函数，根据频率概率和log函数计算
    numEntries = len(dataset) # 总长度做分母
    labelCounts = {} # 利用字典统计各个类别出现的次数
    for featVec in dataset:
        currentLabel = featVec[-1] #类别分类值
        '''if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1'''
        labelCounts[currentLabel] = labelCounts.setdefault(currentLabel, 0) + 1 # 类别计数器
        shannonEnt = 0.0 # 初始化熵为0
    for key in labelCounts: #
        prob = float(labelCounts[key])/numEntries
        print(key,'->',labelCounts[key],':',prob)
        shannonEnt -= prob*math.log(prob, 2)

    return shannonEnt


def splitDataSet(dataset:ndarray,axis:int,value:float) -> ndarray:
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataset.append(reduceFeatVec)
    return retDataset



if __name__ == '__main__':
    daset,labels = createdataset()
    print(daset)
    print(labels)
    shan=calcShannonEnt(daset)
    print(shan)