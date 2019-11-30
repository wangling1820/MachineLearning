"""
 @Author: wangling 
 @Date: 2019-11-30 17:17:13 
 @Last Modified by: wangling
 @Last Modified time: 2019-11-30 17:54:02
"""
import os
import numpy as np

from knn_utils import classify0, classifiedByCosineDis

def get_data(path='./data/testDigits'):
    data_features = []
    data_labels = []
    file_list = os.listdir(path)
    for file in file_list:
        data_labels.append(int(file.split('_')[0]))
        file_path = os.path.join(path, file)
        data = []
        with open(file_path, 'r') as file_data:
            for line in file_data.readlines():
                data.extend([float(i) for i in line.strip()])
        data_features.append(data)
    # print(np.array(data_features).shape)
    return np.array(data_features), data_labels

def number_class_test():
    K = 10
    train_features, train_labels = get_data(path='./data/trainingDigits')
    test_features, test_labels = get_data(path='./data/testDigits')
    error_count_by_eucl = 0.0
    error_count_by_cos = 0.0
    num_test = len(test_labels)
    for index in range(num_test):
        classified_label1 = classify0(test_features[index], train_features, train_labels, K)
        # print('使用欧式距离分出的类别是', classifiedLabel1, ' 实际类别是', test_labels[index])
        if classified_label1 != test_labels[index]:
            error_count_by_eucl += 1
        classified_label2 = classifiedByCosineDis(test_features[index], train_features, train_labels, K)
        # print('使用余弦距离分出的类别是', classifiedLabel2, ' 实际类别是', test_labels[index])
        if classified_label2 != test_labels[index]:
            error_count_by_cos += 1
    print('欧式距离分类错误率为:' , error_count_by_eucl / float(num_test), end=' ')
    print('欧式距离分类正确率为:' , 1- (error_count_by_eucl / float(num_test)))
    print('余弦距离分类错误率为:' , error_count_by_cos / float(num_test), end=' ')
    print('余弦距离分类正确率为:' , 1 - (error_count_by_cos / float(num_test)))


if __name__ == '__main__':
    number_class_test()
