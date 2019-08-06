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
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset,labels

def calcShannonEnt(dataset:ndarray) -> float :
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]




if __name__ == '__main__':
    daset,labels = createdataset()
    print(daset)
    print(labels)