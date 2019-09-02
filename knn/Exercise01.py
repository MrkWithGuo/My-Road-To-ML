#learning from 'Machine Learning in Action' Chapter 2
import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[0.2,0.1],[1.1,1.0],[0,0],[0,0.1],[0.1,0]])
    labels = np.array(['A','B','A','B','B','B'])
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

group,labels=createDataSet()
classify0([0.9,0.8],group,labels,3)
