import numpy as np
import faiss


def L2(l1, l2):
    dist = (l1[0]-l2[0])*(l1[0]-l2[0]) + (l1[1]-l2[1])*(l1[1]-l2[1]) + (l1[2]-l2[2])*(l1[2]-l2[2])
    print(dist)


if __name__ == "__main__":

    d = 3                                           # 向量维度
    nb = 10                                      # index向量库的数据量
    nq = 4                                       # 待检索query的数目
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 5.                # index向量库的向量
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 5.                # 待检索的query向量

    index = faiss.IndexFlatL2(d)
    # print(index.is_trained)         # 输出为True，代表该类index不需要训练，只需要add向量进去即可
    index.add(xb)                   # 将向量库中的向量加入到index中
    # print(index.ntotal)             # 输出index中包含的向量总数，为100000

    print("The xb is:\n{}".format(xb))
    print("The xq is:\n{}\n".format(xq))

    k = 3  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的db索引list，D为其对应的距离(L2 没开平方)
    print("The Index list is:\n{}".format(I[:nq]))     # 前5个最近的
    print("The dist list is:\n{}\n".format(D[-nq:]))    # 也是后5个L2距离最小的

    for ind in range(10):
        l1 = xq[0, :]
        l2 = xb[ind, :]
        print("Now the index is {}".format(ind))
        L2(l1, l2)
