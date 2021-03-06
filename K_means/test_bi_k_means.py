"""
 * @Author: wangling 
 * @Date: 2019-12-25 21:25:27 
"""
from matplotlib import pyplot as plt
from utils import load_dataset, bi_K_means

def plotDataSet(filename, k):
    # 导入数据
    datMat = load_dataset(filename)
    # 进行k-means算法其中k为4
    centList, clusterAssment = bi_K_means(datMat, k)
    clusterAssment = clusterAssment.tolist()
    xcord = [[], [], []]
    ycord = [[], [], []]
    datMat = datMat.tolist()
    m = len(clusterAssment)
    for i in range(m):
        if int(clusterAssment[i][0]) == 0:
            xcord[0].append(datMat[i][0])
            ycord[0].append(datMat[i][1])
        elif int(clusterAssment[i][0]) == 1:
            xcord[1].append(datMat[i][0])
            ycord[1].append(datMat[i][1])
        elif int(clusterAssment[i][0]) == 2:
            xcord[2].append(datMat[i][0])
            ycord[2].append(datMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制样本点
    ax.scatter(xcord[0], ycord[0], s=20, c='b', marker='*', alpha=.5)
    ax.scatter(xcord[1], ycord[1], s=20, c='r', marker='D', alpha=.5)
    ax.scatter(xcord[2], ycord[2], s=20, c='c', marker='>', alpha=.5)
    # 绘制质心
    for i in range(k):
        ax.scatter(centList[i].tolist()[0][0], centList[i].tolist()[0][1], s=100, c='k', marker='+', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()
    
if __name__ == '__main__':
    plotDataSet(filename='./data/testSet2.txt', k=3)
