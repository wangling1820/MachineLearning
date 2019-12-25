"""
 * @Author: wangling 
 * @Date: 2019-12-25 15:08:37 
"""
import numpy as np

def load_dataset(file_name):
    data_mat = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            info = line.strip().split('\t')
            flt_line = list(map(float, info))   # 将数据转换成float类型
            # print(flt_line)
            data_mat.append(flt_line)
    return data_mat


def dis_eclud(a,b):
    return np.sqrt(np.sum(np.power(a-b, 2)))

def rand_cent(data_set, k):
    n = np.shape(data_set)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = min(data_set[:, j])
        range_j = float(max(data_set, ))

if __name__ == '__main__':
    data = load_dataset('./data/testSet.txt')
    print(data)
    print(dis_eclud(np.array(data[0]), np.array(data[1])))
                


