"""
 * @Author: wangling 
 * @Date: 2019-12-25 15:08:37 
"""


def load_dataset(file_name):
    data_mat = []
    with open(file_name, 'r') as file:
        for line in file.readline():
            line.strip().split('\t')
            flt_lit = map(float, line)
            data_mat.append(f)
                


