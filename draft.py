import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def t():
    # gf = np.load("/home/alex/02_ML/01_BEVPlace/BEVPlace/loop/ApolloSpace_train/feature_tsne/gf.npy")
    # print("Load finish")
    # X_embedded = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity = 3).fit_transform(gf)
    # print(X_embedded.shape)
    # np.save("/home/alex/02_ML/01_BEVPlace/BEVPlace/loop/ApolloSpace_train/feature_tsne/tsne_n=2.npy", X_embedded)

    fig = plt.figure()

    X_embedded = np.load("/home/alex/02_ML/01_BEVPlace/BEVPlace/loop/ApolloSpace_train/feature_tsne/tsne_n=3.npy")
    # x_nom = normalization(X_embedded[0])
    # y_nom = normalization(X_embedded[1])
    # z_nom = normalization(X_embedded[2])

    ax = plt.subplot(projection='3d')
    ax.set_title('3d_image_show', fontsize=16)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2])
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")

    # ax = plt.subplot()
    # ax.set_title('2d_image_show', fontsize=12)
    # ax.scatter(X_embedded[:, 0], X_embedded[:, 1], cmap='viridis', marker='+', s=40)

    plt.show()


def t2():
    m = np.array([[1, 3, 3],
                  [4, 1, 6],
                  [1, 4, 2],
                  [1, 6, 3]])
    x = [i[0] for i in m]
    y = [i[1] for i in m]
    z = [2, 2, 2, 3]
    z1 = [i[2] for i in m]

    print(x, y, z)
    print(m)
    # 标准
    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # ax.set_title('3d_image_show', fontsize=20)
    # ax.scatter(x, y, z, c='r', marker='+', s=40)
    # ax.scatter(x, y, z1, c='b', marker='*', s=100)
    # ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    # plt.show()

    # 另一种
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.set_title('3d_image_show', fontsize=20)
    # ax.scatter(x, y, z, c='r', marker='+', s=40)
    # ax.scatter(x, y, z1, c='b', marker='*', s=100)
    # ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    # plt.show()

    # 快速
    ax = plt.subplot(projection='3d')
    ax.set_title('3d_image_show', fontsize=20)
    ax.scatter(x, y, z, c='r', marker='+', s=40)
    ax.scatter(x, y, z1, c='b', marker='*', s=100)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.show()


if __name__ == "__main__":
    t()
