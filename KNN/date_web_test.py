import numpy as np
from knn_utils import classify0, classifiedByCosineDis, autoNorm


def file2matrix(filename='./data/datingTestSet2.txt'):
    fr = open(filename)
    data = fr.readlines()
    dataLen = len(data)
    dataMat = np.zeros((dataLen, 3))
    classLabelsVector = []
    for index, line in zip(range(dataLen), data):
        line = line.strip()
        line2list = line.split('\t')
        classLabelsVector.append(int(line2list[-1]))
        dataMat[index, :] = line2list[:-1]
    # print(classLabelsVector.__class__)
    # print(dataMat.__class__)
    return dataMat, classLabelsVector


def datingClassTest():
    hoRatio = 0.30
    K = 10
    dataMat, classLabels = file2matrix()
    dataMatNorm = autoNorm(dataMat)
    numOtest = int(dataMat.shape[0] * hoRatio)
    errorCountByEucl = 0.0
    errorCountByCos = 0.0
    # 在进行测试的时候,书上将前numOtest个数据作为测试集,将后面的作为训练集.
    for i in range(numOtest):
        classifiedLabel1 = classify0(dataMatNorm[i, :], dataMatNorm[numOtest:, :], classLabels[numOtest:], K)
        print('使用欧式距离分出的类别是', classifiedLabel1, ' 实际类别是', classLabels[i])
        if classifiedLabel1 != classLabels[i]:
            errorCountByEucl += 1
        classifiedLabel2 = classifiedByCosineDis(dataMatNorm[i, :], dataMatNorm[numOtest:, :], 
                    classLabels[numOtest:], K)
        print('使用余弦距离分出的类别是', classifiedLabel2, ' 实际类别是', classLabels[i])
        if classifiedLabel2 != classLabels[i]:
            errorCountByCos += 1
    print('欧式距离分类错误率为:' , errorCountByEucl / float(numOtest))
    print('余弦距离分类错误率为:' , errorCountByCos / float(numOtest))

if __name__ == "__main__":
    # dataMat, classLabelsVector = file2matrix()
    # dataNorm = autoNorm(dataMat)
    # print(dataMat)
    print('-------------')
    # print(dataNorm)
    datingClassTest()

    