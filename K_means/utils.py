"""
 * @Author: wangling 
 * @Date: 2019-12-25 15:08:37 
"""
import numpy as np
from matplotlib import pyplot as plt


def load_dataset(file_name):
    data_mat = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            info = line.strip().split('\t')
            flt_line = list(map(float, info))   # 将数据转换成float类型, float(info[0]) float(info[1])
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


def bi_K_means(dataset, k, dist_meas=dist_eclud):
    m = dataset.shape[0]
    cluster_assment = np.mat(np.zeros((m,2)))
    centroid_0 = np.mean(dataset, axis=0).tolist()[0] # 将全部数据的均值作为整体簇的中心点
    cent_list = []
    cent_list.append(centroid_0) # 保存第一个数据中心点
    for j in range(m): # 存储分类信息和损失(也每个点到中心点的距离)
        cluster_assment[j, 1] = dist_meas(np.mat(centroid_0), dataset[j, :]) ** 2
    # 将数据进行再次聚类,直到满足要求. sse is 'sum of squared error'
    while (len(cent_list) < k):
        print(len(cent_list))
        lowest_sse = float('inf')
        for i in range(len(cent_list)):
            pts_in_curr_cluster = dataset[np.nonzero(cluster_assment[:, 0].A==i)[0], :]
            if len(pts_in_curr_cluster) == 0:  # 有时会出现该重心周围没有数据的情况
                continue
            centroid_mat, split_clust_ass = K_means(pts_in_curr_cluster, 2, dist_meas) # 第一个是分类,第二个是损失.
            sse_split = np.sum(split_clust_ass[:, 1]) # 统计所有数据距离中心点的距离
            sse_not_split = np.sum(cluster_assment[np.nonzero(cluster_assment[:, 0].A!=i)[0], 1]) # 统计不分割时的距离
            print('see_split is ', sse_split, '  sse_not_split is: ', sse_not_split, '  lowest_sse is ', lowest_sse)
            if (sse_split + sse_not_split) < lowest_sse: # 比较不同拆分方式的损失大小
                best_cent_to_split = i # 再次分割的中点
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass.copy()
                lowest_sse = sse_split + sse_not_split

        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A==0)[0], 0] = best_cent_to_split    # 更新再次聚类后的第一个簇的重心位置的idx
        cent_list[best_cent_to_split] = best_new_cents[0, :]    # 更新再次聚类的中心位置

        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A==1)[0], 0] = len(cent_list)    # 更新再次聚类后的第二个簇的重心位置idx
        cent_list.append(best_new_cents[1, :])  # 更新再次聚类的中心位置

        print('the best_cent_to_split is: ', best_cent_to_split)
        print('the len of best_clust_ass is: ', len(best_clust_ass))

        # 在结果中进行更新
        cluster_assment[np.nonzero(cluster_assment[:, 0].A==best_cent_to_split)[0], :] = best_clust_ass
    return cent_list, cluster_assment


# test function
if __name__ == '__main__':
    # 测试load_dataset
    data = load_dataset('./data/testSet.txt')
    print(data)
    print(dist_eclud(data[0], data[1]))
    print(rand_cent(data, 4))
    # 测试K_means
    cent, cluster = K_means(data, 4)
    print(cent)
    print(cluster)
    # 测试bi_K_means
    data_1 = load_dataset('./data/testSet2.txt')
    cent_1, cluster_1 = bi_K_means(data_1, 3)
    print(cent_1)
    print(cluster_1)

