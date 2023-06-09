import os.path

import numpy as np
import faiss

from tqdm import tqdm

# 定义全局变量
d = 3  # 向量维度
nb = 10  # index向量库的数据量
nq = 4  # 待检索query的数目
thre = 5


def L2(l1, l2):
    dist = (float(l1[0])-float(l2[0]))*(float(l1[0])-float(l2[0])) \
           + (float(l1[1])-float(l2[1]))*(float(l1[1])-float(l2[1])) \
           + (float(l1[2])-float(l2[2]))*(float(l1[2])-float(l2[2]))
    return dist


def data_gen():
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 5.                # index向量库的向量
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 5.                # 待检索的query向量
    return xq, xb


def search(xq, xb):
    index = faiss.IndexFlatL2(d)
    # print(index.is_trained)         # 输出为True，代表该类index不需要训练，只需要add向量进去即可
    index.add(xb)                   # 将向量库中的向量加入到index中
    # print(index.ntotal)             # 输出index中包含的向量总数，为100000

    k = 3  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的db索引list，D为其对应的距离(L2 没开平方)
    # print("The Index list is:\n{}".format(I[:nq]))     # 前5个最近的
    # print("The dist list is:\n{}\n".format(D[-nq:]))    # 也是后5个L2距离最小的
    return D, I


def val_search(xq, xb, q_s, q_e):
    # 验证各个排名
    L2_vector=[]
    for ind in range(len(xb)):
        db_info = [1, 2]
        if ind >= q_s and ind <= q_e:
            continue
        l1 = xq
        l2 = xb[ind]
        # print("Now the index is {}".format(ind))
        L2_dist = L2(l1, l2)
        if L2_dist < thre:
            db_info[0] = ind
            db_info[1] = L2_dist
            L2_vector.append(db_info)
    return L2_vector
    # 相当于对于list的下标集合（即“range(len(L2_vector))”）
    # 按照规则以list列表中的元素（key=lambda k:L2_vector[k]，lambda是匿名函数的意思）
    # 升降序（reverse=False）进行排列
    # sorted_id = sorted(range(len(L2_vector)), key=lambda x: L2_vector[x], reverse=False)
    # return sorted_id[:3]


def data_load(path, filename):

    file_path = os.path.join(path, filename)
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

    return pcd_index, pcd_location


def query_gen(loc):
    L = len(loc)
    query_info = []

    print("query index generation:")
    with tqdm(total=len(loc)) as t:
        for i in range(L):
            query_index = list(range(2))
            flag_1 = True
            flag_2 = True

            k = 0
            while flag_1:
                l1 = loc[i]
                l2 = loc[i-k]
                if L2(l1, l2) > thre or i-k <= 0:
                    query_index[0] = i-k  # 向前5m
                    flag_1 = False
                k = k + 1
            j = 0
            while flag_2:
                l1 = loc[i]
                l2 = loc[i+j]
                if L2(l1, l2) > thre or i+j >= L-1:
                    query_index[1] = i+j  # 向后5m
                    flag_2 = False
                j = j + 1
            t.update(1)
            query_info.append(query_index)

    save_list(query_info, "query_info")

    return query_info


def save_list(list1, filename):
    file_path = os.path.join('./loop', filename)
    file = open(file_path + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file.write('\t')                          # 相当于Tab一下，换一个单元格
        file.write('\n')                              # 写完一行立马换行
    file.close()


def loop_detect(info, loc):

    query_loop = []
    db = loc
    # db = np.array(db).reshape(len(loc), 3)
    print("close loop detection")
    with tqdm(total=len(loc)) as t:
        for i in range(len(loc)):
            query = loc[i]
            # query = np.array(query).reshape(1, 3)
            q_s = info[i][0]
            q_e = info[i][1]
            loop = val_search(query, db, q_s, q_e)
            query_loop.append(loop)
            t.update(1)

    save_list(query_loop, "query_loop")
    return query_loop


if __name__ == "__main__":

    data_path = "/home/alex/Dataset/Apollo/SanJoseDowntown_TrainData/SanJoseDowntown/2018-10-11/poses"
    file_name = "gt_poses.txt"

    index, location = data_load(data_path, file_name)
    info = query_gen(location)

    loop_result = loop_detect(info, location)
    print(len(loop_result))
