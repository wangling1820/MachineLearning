"""
 * @Author: wangling 
 * @Date: 2019-12-25 21:48:41 
"""
from math import sin, cos, pi, acos
import numpy as np
from matplotlib import pyplot as plt

from utils import bi_K_means


def load_dataset(file_name='./data/places.txt'):
    data = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            info = line.strip().split('OR')[-1].strip().split('\t')
            data.append(list(map(float, info)))
    return np.mat(data)


def dist_slc(a, b):
    x = sin(a[0, 1] * pi / 180) * sin(b[0, 1] * pi / 180)
    y = cos(a[0, 1] * pi / 180) * cos(b[0, 1] * pi / 180) * cos(pi * (b[0, 0] - a[0, 0]) / 180)
    return acos(x + y) * 6371.0


def cluster_clubs(dataset, num_clust=6):
    my_centroids, clust_assing = bi_K_means(dataset, num_clust, dist_meas=dist_slc)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatter_markers = ['d', 'v', 's', 'o', '^', '8', 'p', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax_0 = fig.add_axes(rect, label='ax0', **axprops)
    img_ground = plt.imread('./data/Portland.png')
    ax_0.imshow(img_ground)
    ax_1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(num_clust):
        pts_in_curr_cluster = dataset[np.nonzero(clust_assing[:, 0].A==i)[0], :]
        print(pts_in_curr_cluster)
        if len(pts_in_curr_cluster) == 0:
            continue
        marker_style = scatter_markers[i % len(scatter_markers)]
        ax_1.scatter(pts_in_curr_cluster[:, 0].flatten().A[0], pts_in_curr_cluster[:, 1].flatten().A[0],
            marker=marker_style, s=90)
    for i in range(num_clust):
        ax_1.scatter(my_centroids[i].tolist()[0][0], my_centroids[i].tolist()[0][1], s=300, c='k', marker='+', alpha=.5)

    plt.show()


if __name__ == "__main__":
    # test load_dataset
    data = load_dataset()
    print(data)
    # test dist_slc
    print(dist_slc(data[0], data[1]))
    cluster_clubs(data)
    





