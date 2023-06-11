import dataset
import os
import numpy as np
import matplotlib.pyplot as plt

# def data_load(path, filename):
#
#     file_path = os.path.join(path, filename)
#     pcd_index = []
#     pcd_location = []
#
#     try:
#         file = open(file_path, 'r')
#     except FileNotFoundError:
#         print('File is not found')
#     else:
#         lines = file.readlines()
#         for line in lines:
#             position = list(range(3))
#             a = line.split()
#             index = a[0]
#             position[0] = a[2]
#             position[1] = a[3]
#             position[2] = a[4]
#
#             pcd_index.append(index)
#             pcd_location.append(position)
#     file.close()
#
#     return pcd_index, pcd_location
#
#
# if __name__ == '__main__':
#
#     data_path = "/home/alex/Dataset/Apollo"
#     file = 'gt_poses.txt'
#
#     _, position = data_load(data_path, file)
#     position = np.array(position)
#     print("\nThe range of x is:{} ~ {}".format(min(position[0]), max(position[1])))
#     print("\nThe range of y is:{} ~ {}".format(min(position[1]), max(position[1])))
#     print("\nThe length of data is:{}".format(len(position)))


x = np.linspace(0,10,55)

plt.plot(x,np.sin(x-0),'o',label='o',markerfacecolor='r',markersize=5,markeredgecolor='gray',markeredgewidth=2)

# plt.legend(numpoints=6,loc='right') # 改变numpoints看看变化就知道其作用
# plt.xlim(0, 12)
plt.show()
