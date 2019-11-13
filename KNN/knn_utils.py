import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels =['A', 'A', 'B', 'B']
    return group, labels


# 使用欧式距离计算相似度,书上采用的是这种方法
def classify0(intX, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    testData = np.tile(intX, (dataSize, 1)) # 将数据扩展成矩阵形式,便于使用矩阵运算
    # print(testData)
    # print(intX)
    diffMat = testData - dataSet
    diffMat = diffMat ** 2
    # print(diffMat.__class__) <numpy.ndarray>
    diffMat = diffMat.sum(axis=1)  # 1是行 0是列 按行将数据求和
    distance = diffMat ** 0.5
    sortedDisIndices = distance.argsort()   
    classCount = {}
    for i in range(k):
        classLabel = sortedDisIndices[i]
        # classLabel = labels[sortedDisIndices[i]]  # 书上
        classCount[classLabel] = classCount.get(classLabel, 0) + 1
    # sorted(iterable, cmp=None, key=None, reverse=False) 使用key可以取代传入比较函数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return labels[sortedClassCount[0][0]]
    # return sortedClassCount[0][0] # 书上



# 使用余弦距离作为相似度度量方法
# cosine = x * y / (||x|| * ||y||)
# 将相似度计算中分母加1,然后将计算的相似度乘以了-1.
def classifiedByCosineDis(inX, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    testData = np.tile(inX, (dataSize, 1))
    testDataNorm = (testData**2).sum(axis=1)
    dataSetNorm = (dataSet**2).sum(axis=1)
    # print(1/testDataNorm)
    # print(dataSetNorm)
    cosineDis = -1 * ((testData * dataSet).sum(axis=1) / (testDataNorm * dataSetNorm + 1))  
    # 此处余弦相似度乘以-1是为了排序方便；分母加1为了解决分母出现0的情况,其中加1操作不会影响到最终的结果.
    sortedCosineDis = cosineDis.argsort()
    classCount = {}
    for i in range(k):
        classLabel = labels[sortedCosineDis[i]]
        classCount[classLabel] = classCount.get(classLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 对数据进行归一化处理,统一不同数据信息的范围
# 归一化的操作为 (value - min) / (max - min)
def autoNorm(dataSet):
    # ndarray.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
    dataMin = dataSet.min(0)    
    dataMax = dataSet.max(0)    
    dataRange = dataMax - dataMin   # 此时的维度与原始数据不同,需要进行惟独的扩展
    dataRange = np.tile(dataRange, (dataSet.shape[0], 1))
    dataMin = np.tile(dataMin, (dataSet.shape[0], 1))
    dataSetNorm = (dataSet - dataMin) / dataRange
    print(dataSetNorm)
    return dataSetNorm


def test1():
    group, labels = createDataSet()
    label = classify0([1,1], group, labels, 3)
    return label

if __name__ == "__main__":
    # 测试creatrDataSet()
    group, labels = createDataSet()
    
    print(group.shape, "\n" , group, "\n", labels)
    print('-----------------------------------')
    # 测试classify0函数
    print(classify0([1, 1], group, labels, 4))
    print('-----------------------------------')
    # 测试calssfiedByCosineDis
    print(classifiedByCosineDis([1,1], group, labels, 4))
    # 测试test01函数
    print(test1())
