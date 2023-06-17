import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os

data_path = "/home/alex/02_DL/02_BEVPlace/BEVPlace/data/Apollo/SanJose_train"
file = 'gt_poses.txt'

file_path = os.path.join(data_path, file)
pcd_index = []
pcd_location = []
try:
    file = open(file_path, 'r')
except FileNotFoundError:
    print('File is not found')
else:
    lines = file.readlines()
    for line in lines:
        position = list(range(3))
        a = line.split()
        index = a[0]
        position[0] = a[2]
        position[1] = a[3]
        position[2] = a[4]

        pcd_index.append(index)
        pcd_location.append(position)
    file.close()

pcd_location = np.array(pcd_location)
x = pcd_location[:10, 0]
y = pcd_location[:10, 1]

plt.scatter(x, y)
plt.xticks(range(len(x)), x)
plt.show()




# def init():
#     ax.set_xlim(min(pcd_location[0]), max(pcd_location[0]))
#     ax.set_ylim(min(pcd_location[1]), max(pcd_location[1]))
#     # ax.set_xlim(-40, 590000)
#     # ax.set_ylim(-40, 590000)
#     # ax.set_xlim(-np.pi, np.pi)
#     # ax.set_ylim(-1, 1)
#     print("1")
#     return line,


# def update(frame):
#     # x.append(pcd_location[0][frame])
#     # y.append(pcd_location[1][frame])
#     # x.append(frame)
#     # y.append(np.sin(frame))
#     # line.set_data(x, y)
#     if len(ax.lines) == 2:
#         ax.lines.pop(1)
#     ax.plot(x[frame], y[frame], 'o', color='red')
#     print("2")
#     return line, ax
#
#
# ani = FuncAnimation(fig, update
#                    , frames=len(x)
#                    , interval=10
#                    # , init_func=init
#                    , blit=True)
# plt.show()
# ani.save("animation.gif", fps=50, writer="imagemagick")
