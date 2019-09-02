#学习《机器学习实战》（Machine Learning in Action）第2章的代码  

>import numpy as np  
>import operator  

>def createDataSet():  
>>    #原书数据较少，现增加2条，以体现k近邻的思想  
>>    group=np.array([[1.0,1.1],[0.2,0.1],[1.1,1.0],[0,0],[0,0.1],[0.1,0]])  
>>    labels=np.array(['A','B','A','B','B','B'])  
>>    #group  labels  每次循环执行时的xB0和xB1  
>>    #[1.0,1.1]  ['A']   xB0=1.0,xB1=1.1  
>>    #[0.2,0.1]  ['B']   xB0=0.2,xB1=0.1  
>>    #[1.1,1.0]  ['A']   xB0=1.1,xB1=1.0  
>>    #[0 , 0  ]  ['B']   xB0=0  ,xB1=0  
>>    #[0  ,0.1]  ['B']   xB0=0  ,xB1=0.1  
>>    #[0,1,0  ]  ['B']   xB0=0.1,xB1=0  
>>    return group,labels  

>def classify0(inX,dataSet,labels,k):  
>>    #inX是输入数据(最后1行的输入值，可以自行修改)  
>>    #dataSet是训练数据集(第8行的group)  
>>    #labels是分类标签集(第9行的labels)  
>>    #k是最近邻居的数量  

>>    #取行数,结果是6(行)
>>    dataSetSize=dataSet.shape[0]  
 
>>    #xA0和xA1对应inX的第1列和第2列的值,本例的输入值inX=[0.9,0.8]  
>>    #xA0=0.9,xA1=0.8,只有1行数据  
>>    #[xB0,xB1]则有6行数据,需要将[xA0,xA1]也变成6行,便于进行矩阵运算[6行2列]-[6行2列]=[6行2列]
>>    #np.tile(inX,(dataSetSize,1))代码的tile函数，将inX按行

>>    #![image](https://github.com/MrkWithGuo/My-Road-To-ML/blob/master/knn/images/Exercise01_01.gif) 
>>    diffMat=np.tile(inX,(dataSetSize,1))-dataSet  
>>    #[(xA0-xB0)**2,(xA1-xB1)**2]  
>>    sqDiffMat=diffMat**2  
>>    #.sum不加参数所有相加；axis=1按行相加；axis=0按列相加  
>>    #(xA0-xB0)**2+(xA1-xB1)**2  
>>    sqDistances=sqDiffMat.sum(axis=1)  
>>    #(xA0-xB0)**2+(xA1-xB1)**2)**0.5
>>    distances=sqDistances**0.5  
>>    #从小到大排序的索引值  
>>    sortedDistIndicies=distances.argsort()  
>>    #看看结果
>>    print(distances)  
>>    print(sortedDistIndicies)  
>>    classCount={}  
>>    #k近邻  
>>    for i in range(k):  
>>>        #按排序后labels值  
>>>        voteIlabel=labels[sortedDistIndicies[i]]  
>>>        #看看过程
>>>        print(dataSet[sortedDistIndicies[i]])  
>>>        print(voteIlabel)  
>>>        #根据不同的labels值，用字典进行总数统计  
>>>        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  
>>>        print(classCount[voteIlabel])  
>>    print(classCount)  
>>    #itemgetter(1)对字典的值排序；（0）对字典的键排序  
>>    #按照labels总数进行排序，比如(A:2),(B:1)，说明结果更接近A  
>>    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  
>>    print(sortedClassCount)  
>>    #返回预测的最近结果  
>>    return sortedClassCount[0][0]  

>group,labels=createDataSet()
>classify0([0.9,0.8],group,labels,3)
