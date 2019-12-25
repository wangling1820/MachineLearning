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
    # return data_mat
    return np.array(data_mat)


def dist_eclud(a,b):
    return np.sqrt(np.sum(np.power(a-b, 2)))


def rand_cent(data_set, k):
    n = data_set.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = np.min(data_set[:, j]) # 找该维度最小的
        range_j = float(np.max(data_set[:, j]) - min_j) # 找该维度变换范围
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1) 
        # 随机生成该维度的中心坐标,起点+长度*0.x return random values in a given shape-[k, 1].
    return centroids


def K_means(dataset, k, dist_meas=dist_eclud, create_cent=rand_cent):
    m = dataset.shape[0] # 数据总数
    cluster_assment = np.mat(np.zeros((m, 2))) # 存储每个数据的中心位置和距离
    centroids = create_cent(dataset, k) # 初始化重心
    cluster_changed = True  # flag
    while cluster_changed:
        cluster_changed = False
        for i in range(m):  # 遍历每个数据, 为每个数据分配一个重心
            min_dist = float('inf') # 初始化距离
            min_idx = -1
            for j in range(k): # 遍历每个重心
                dist_ij = dist_meas(dataset[i], centroids[j])
                if dist_ij < min_dist:
                    min_dist = dist_ij
                    min_idx = j
                if cluster_assment[i, 0] != min_idx:
                    cluster_changed = True
                    cluster_assment[i, 0] = min_idx
                    cluster_assment[i, 1] = min_dist**2
        # 更新重心的位置, 使用每个簇的中心点代替重心
        # print(centroids)
        for cent in range(k):
            # 筛选出该重心周围的点
            # nonzero: Return the indices of the elements that are non-zero.
            # print(np.nonzero(cluster_assment[:, 0].A==cent)[0])
            pts_in_clust = dataset[np.nonzero(cluster_assment[:, 0].A==cent)[0]]
            # print(pts_in_clust)
            # 计算该簇所有数据的中心点,并更新重心
            centroids[cent, :] = np.mean(pts_in_clust, axis=0)
    return centroids, cluster_assment


def biKmeans(dataset, k, dist_meas=dist_eclud):
    m = dataset.shape[0]
    cluster_assment = np.mat(np.zeros((m,2)))
    centroid_0 = np.mean(dataset, axis=0).tolist()[0]
    cent_list = [centroid_0]
    for j in range(m):
        cluster_assment[j, 1] = dis_meas()

# test function
if __name__ == '__main__':
    data = load_dataset('./data/testSet.txt')
    print(data)
    print(dist_eclud(data[0], data[1]))
    print(rand_cent(data, 4))
    cent, cluster = K_means(data, 4)
    print(cent)
    print(cluster)


