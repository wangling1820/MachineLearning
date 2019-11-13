import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels =['A', 'A', 'B', 'B']
    return group, labels
    pass


# 使用欧式距离计算相似度
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
    # sorted(iterable, cmp=None, key=None, reverse=False)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return labels[sortedClassCount[0][0]]
    # return sortedClassCount[0][0] # 书上
    pass


def test1():
    group, labels = createDataSet()
    label = classify0([1,1], group, labels, 3)
    return label
    pass
if __name__ == "__main__":
    # 测试creatrDataSet()
    group, labels = createDataSet()
    
    print(group.shape, "\n" , group, "\n", labels)
    print('-----------------------------------')
    # 测试classify0函数
    print(classify0([1, 1], group, labels, 4))
    print('-----------------------------------')
    # 测试test01函数
    print(test1())
