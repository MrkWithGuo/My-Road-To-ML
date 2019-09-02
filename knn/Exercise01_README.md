#学习《机器学习实战》（Machine Learning in Action）第2章的代码  

import numpy as np  
import operator  

def createDataSet():  
    >>group=np.array([[1.0,1.1],[0.2,0.1],[1.1,1.0],[0,0],[0,0.1],[0.1,0]])  
    >>labels=np.array(['A','B','A','B','B','B'])  
    >>return group,labels  

def classify0(inX,dataSet,labels,k):  
    >>dataSetSize=dataSet.shape[0]#取行数  
    >>diffMat=np.tile(inX,(dataSetSize,1))-dataSet#[(xA0-xB0),(xA1-xB1)]  
    >>sqDiffMat=diffMat**2#[(xA0-xB0)**2,(xA1-xB1)**2]  
    >>#.sum不加参数所有相加；axis=1按行相加；axis=0按列相加  
    >>sqDistances=sqDiffMat.sum(axis=1)#(xA0-xB0)**2+(xA1-xB1)**2  
    >>distances=sqDistances**0.5#(xA0-xB0)**2+(xA1-xB1)**2)**0.5  
    >>sortedDistIndicies=distances.argsort()#从小到大排序的索引值  
    >>print(distances)  
    >>print(sortedDistIndicies)  
    >>classCount={}  
    >>for i in range(k):#k近邻  
        >>>>voteIlabel=labels[sortedDistIndicies[i]]#按排序后labels值  
        >>>>print(dataSet[sortedDistIndicies[i]])  
        >>>>print(voteIlabel)  
        >>>>classCount[voteIlabel]=classCount.get(voteIlabel,0)+1#根据不同的labels值，用字典进行总数统计  
        >>>>print(classCount[voteIlabel])  
    >>print(classCount)  
    >>#itemgetter(1)对字典的值排序；（0）对字典的键排序  
    >>#按照labels总数进行排序，比如(A:2),(B:1)，说明结果更接近A  
    >>sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  
    >>print(sortedClassCount)  
    >>#返回预测的最近结果  
    >>return sortedClassCount[0][0]  

group,labels=createDataSet()  
classify0([0.9,0.8],group,labels,3)  
