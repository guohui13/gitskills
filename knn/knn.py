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
import os


# 创建一个数据集和标注
def createDataSet() -> 'ndarray,list':
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
    sqDistances = sqDiffMat.sum(axis=1)  # 朝着变换的下标来
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
def file2matrix(filename: str) -> ndarray:
    dic = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}  # 分类标签的字符和数字的同一描述
    try:
        with open(filename) as fr:
            arrayOlines = fr.readlines()  # 文件缓存
            numberOfLines = len(arrayOlines)  # 文件长度
            returnMat = zeros((numberOfLines, 3))  # 构造0矩阵
            classLabelVector = []
            index = 0

            for line in arrayOlines:  # line 是str类型
                line = line.strip()  # line 仍是str 类型
                listFromLine = line.split('\t')  # line转换成list
                returnMat[index, :] = listFromLine[0:3]  # 矩阵赋值，将list转换成ndarray格式
                if listFromLine[-1].isdigit():
                    classLabelVector.append(int(listFromLine[-1]))
                else:
                    classLabelVector.append(int(dic.get(listFromLine[-1])))
                index += 1
            return returnMat, classLabelVector
    except Exception as e:
        print(e)


# 图形化数据进行数据可视化分析
def analysis_data(groups: ndarray, labels: ndarray) -> None:
    try:
        datingDataMat, labels = groups, datingLabels
        plt.figure(1)  # pyplot有图形和轴的概念 figure代表当前图形 图形1
        ax1 = plt.subplot(221)  # 图形划分四块使用第一块
        # ax 画散点图 坐标 x，y ，点大小，颜色
        ax1.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels),
                    15.0 * array(datingLabels))  # 两列数据，三个分类
        ax2 = plt.subplot(222)  # 图形划分四块使用第2块
        ax2.scatter(datingDataMat[:, 0], datingDataMat[:, 2], 15.0 * array(datingLabels),
                    15.0 * array(datingLabels))  # 两列数据，三个分类
        ax3 = plt.subplot(223)
        ax3.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels),
                    15.0 * array(datingLabels))  # 两列数据，三个分类
        ax4 = plt.subplot(224)
        ax4.scatter(datingDataMat[:, 1], datingDataMat[:, 2])  # 原始图像不加入颜色的时候
        plt.show()
    except Exception as e:
        print(e)


# 归一化矩阵 (oldValue-min)/(max-min)
def autoNorm(dataSet: ndarray) -> ndarray:
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet)) # 使用0阵初始化一个矩阵
    m = dataSet.shape[0]
    normson = dataSet - tile(minVals, (m, 1))  # tile 复制多行
    # normmom = tile(maxVals,(m,1)) - tile(minVals,(m,1)) #复制多行
    normmom = tile(ranges, (m, 1))
    # python 做线性代数运算的时候，两种运算，一种叫 element-wise的叫逐个元素计算 ，还有一种是矩阵运算
    norm = normson / normmom  # element-wise
    return norm, ranges, minVals


# 测试分类器效果
def datingClassTest() -> None:
    hoRatio = 0.10  # 设置抽样测试比例
    datingDataMat, datingLabels = file2matrix(r'./data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 用前10%的数据作为输入数据点， 用90%的数据做数据集 ，得到分类结果
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],
                                     5)  # k为偶数分类器错误率提升
        print("分类器分类结果为 : {:.0f} , " \
              "实际结果为 : {:.0f} ".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            print("↑这个分类错误↑")
            errorCount += 1
    print("the total error rate is : {:.2%}".format(errorCount / float(numTestVecs)))


# 手动输入数据产生分类结果
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))  # python3 不再使用raw_input
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix(r'./data/datingTestSet.txt')
    normat, ranges, minvals = autoNorm(datingDataMat)
    in_data = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(in_data / ranges, normat, datingLabels, 5)
    print("you will probably like this person : {}".format(resultList[classifierResult - 1]))  # 得到的分类标签跟数组差1 所以减法


def img2vector(filename: str) -> ndarray:
    vec = []
    with open(filename) as fr:
        for line in fr.readlines():
            vec.append(list(line.strip()))
    ndvec = array(vec, dtype=float64)
    ndvec1 = ndvec.reshape(1, 1024)
    return ndvec1



def handwriteing():
    hwlabels=[]
    trainfiles = os.listdir('./data/trainingDigits')
    tcnt = len(trainfiles)
    # 这个书里的矩阵转换都是使用zeros来初始化的，个人采用的np.array转list成array不如这个易于阅读
    trainingmat = zeros((tcnt,1024))
    for i in range(tcnt): # 训练数据赋值和标签分类
        filename = trainfiles[i]
        filestr = filename.split('.')[0]
        classnum = int(filestr.split('_')[0])
        hwlabels.append(classnum)
        trainingmat[i] = img2vector('./data/trainingDigits/'+filename)
    testfiles = os.listdir('./data/testDigits/')
    errcount=0.0
    testcnt = len(testfiles)
    for i in range(testcnt): # 循环内逐个测试
        filename = testfiles[i]
        filestr = filename.split('.')[0]
        classnum = int(filestr.split('_')[0])
        vect_test = img2vector('./data/testDigits/' + filename)
        test_classnum = classify0(vect_test,trainingmat,hwlabels,5)
        print("分类器结果 {:d}，实际结果 {:d}".format(test_classnum,classnum) )
        if classnum != test_classnum:
            errcount += 1
    print("分类器的准确率是 {:.2%}".format(float((testcnt - errcount)/testcnt)))






if __name__ == '__main__':
    groups, labels = createDataSet()
    k = classify0([1, 1], groups, labels, 3)
    print("iput belongs class {}".format(k))
    datingDataMat, datingLabels = file2matrix(r'./data/datingTestSet.txt')
    print(datingDataMat)
    print("----------------------datingLabels---------------------")
    print(datingLabels)
    '''
    fig1=plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(datingDataMat[:,1],datingDataMat[:,2])
    plt.show()
    fig2=plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0*array(datingLabels),
               15.0*array(datingLabels))
    plt.show()
    '''
    analysis_data(datingDataMat, datingLabels)  # 分析数据
    datingClassTest()
    #classifyPerson()
    ret=img2vector(r'./data/trainingDigits/8_54.txt')
    handwriteing()

