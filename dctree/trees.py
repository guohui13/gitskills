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
        currentLabel = featVec[-1] #类别分类值，根据分类标签分类
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
    #划分矩阵，返回划分后符合条件的矩阵，axis是划分轴（特征轴），value是划分轴的划分值,不符合条件的值的所在样本（即行）全部刨除，画图易于理解
    retDataset = [] #返回的划分后的矩阵的初始化
    for featVec in dataset: #遍历矩阵行
        if featVec[axis] == value: # 如果划分轴所在的值跟要划分的值对应，就开始拆分样本，否则不保留
            reduceFeatVec = featVec[:axis] # 划分轴的前面元素集合
            reduceFeatVec.extend(featVec[axis + 1:]) # 轴前后集合元素组合
            retDataset.append(reduceFeatVec) # 满足了划分条件后的样本元素构成矩阵
    return retDataset

def chooseBestFeatureTosplit(dataset:ndarray)-> list:
    numFeatures = len(dataset[0]) - 1 # 特征的数量(列数)，去掉分类标签
    baseEntropy = calcShannonEnt(dataset) # 计算未划分时原始数据的熵
    bestInfoGain = 0
    bestFeature = -1 #
    for i in range(numFeatures): #挨个特征的轴划分
        featList = [ example[i] for example in dataset ] # 获取特征列元素，组成列表
        uniqvals = set(featList) # 去重特征轴的列表值
        subEntropy = 0.0 #
        for value in uniqvals: # 根据划分轴的不同元素值进行划分
            subdataset = splitDataSet(dataset, i, value) # 划分轴划分后的结果集
            prob = len(subdataset)/float(len(dataset)) #划分后的数据的分类概率
            subEntropy += prob*calcShannonEnt(subdataset) #信息增益公式的被减数 Gain(S,A)=Entropy(S) - (Sv/S) * Entropy(Sv)
            # 信息增益公式  信息增益（划分前，划分后）= 划分前的熵 - sum(（划分后的数据集的长度/ 划分前的数据集的长度）* 划分后的数据集的熵)
        infoGain = baseEntropy - subEntropy # 信息增益最大化的特征就是最好的特征
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature



if __name__ == '__main__':
    daset,labels = createdataset()
    print(daset)
    print(labels)
    shan=calcShannonEnt(daset)
    print(shan)
    vv=splitDataSet(daset,0,0)
    vv1 = splitDataSet(daset, 0, 1)
    print(vv)
    print(vv1)
    axis = chooseBestFeatureTosplit(daset)
    print("axis is %d",axis)