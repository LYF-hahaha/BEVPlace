import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import open3d as o3d
from tqdm import tqdm
from threading import Thread
from Show_Path_Apollo import TrajGen


def load_check_file(path, name):
    file_path = os.path.join(path, name)
    with open(file_path, 'r') as f:
        content = f.read().split('\n')
        recall_n = []
        query = []
        pair = []
        for i in range(len(content)-1):
            a = content[i].split('\t')
            recall_n.append(int(a[0]))
            query.append(int(a[1]))
            b = []
            for j in a[2].split():
                b.append(int(j))
            pair.append(b)
        f.close()
    return recall_n, query, pair


def load_check_file_np(path, name):
    file_path = os.path.join(path, name)
    result = np.load(file_path)
    return result


# 载入根据gps信息计算的回环数量（总计4638个）
def load_loop_gt(path, name):
    file_path = os.path.join(path, name)
    with open(file_path, 'r') as pair:
        reader = csv.reader(pair)
        pair_result = []
        for row in reader:
            temp = [int(row[0]), int(row[1][1:-1].split(',')[0])]
            pair_result.append(temp)
        pair.close()
    pair_result = np.array(pair_result)

    return pair_result


# pred的正确&错误数量可视化（柱状图）
def recall_bar(n):
    x = [1, 2, 3, 4, 5]
    a, b, c, d, e = acc_cal(n)
    y = [a, b, c, d, e]
    l = ['wrong', 'correct@1', 'correct@5', 'correct@10', 'correct@20']

    plt.figure('Distribution of n')
    plt.bar(x, y, tick_label=l, width=0.4)
    plt.ylim((0, max(a, b, c, d, e)*1.2))

    for i, j in zip(x, y):
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(i, j + 0.05,  # 文字的位置
                 'num=%.2f\n acc=%.2f' %(j, (j/len(n))),  # 文字内容（传进来的浮点数保留两位数字）
                 ha='center', va='bottom')  # 水平&垂直方向的对齐方式

    plt.show()
    plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
             fontdict={'size': 16, 'color': 'r'})


# 统计每个n的数量
def acc_cal(n):
    wrong = 0
    n_1 = 0
    n_5 = 0
    n_10 = 0
    n_20 = 0

    for i in n:
        if i == 0:
            wrong = wrong+1
        if i ==1:
            n_1 = n_1+1
        if i ==5:
            n_5 = n_5+1
        if i ==10:
            n_10 = n_10+1
        if i ==20:
            n_20 = n_20+1
    # print("The pred acc is:\nWrong rate:")
    # wrong_rate = wrong/len(n)
    # acc_1 = n_1/len(n)
    return wrong, n_1, n_5, n_10, n_20


# 载入预测结果文件，统计其中正确、错误的数量（总计2328个）
def wrong_filter(path, name):
    result = load_check_file_np(path, name)
    query = []
    pair = []
    for i in range(len(result)-1):
        if result[i, 0] == 0:
            query.append(result[i, 1])
            pair.append(result[i, 2:])
    return query, pair


def correct_pair(wrong_query, wrong_pair, gt):
    compair_table = []
    for i in range(len(wrong_query)):
        for j in range(gt.shape[0]):
            if wrong_query[i] == gt[j][0]:
                temp = [wrong_query[i], gt[j][1], wrong_pair[i][0], wrong_pair[i][1], wrong_pair[i][2]]
                compair_table.append(temp)
    # wrong_query  correct_pair  wrong_pair*3
    compair_table = np.array(compair_table)

    with open('../../loop/compair_table_10800_11800.csv', 'w') as f:
        writer = csv.writer(f)
        # writer.writerow(["wrong_query", "correct_pair", "w1", "w2", "w3"])
        for i in range(len(compair_table)):
            writer.writerow(compair_table[i])
        f.close()
    return compair_table


def vis_cpt(data_dir, cpt, index):

    vis_q = o3d.visualization.Visualizer()
    vis_c = o3d.visualization.Visualizer()
    vis_w = o3d.visualization.Visualizer()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    vis_q.clear_geometries()
    vis_c.clear_geometries()
    vis_w.clear_geometries()

    pcd_q = o3d.io.read_point_cloud(os.path.join(data_dir, str(cpt[index][0])+'.pcd'))
    pcd_c = o3d.io.read_point_cloud(os.path.join(data_dir, str(cpt[index][1])+'.pcd'))
    pcd_w = o3d.io.read_point_cloud(os.path.join(data_dir, str(cpt[index][2])+'.pcd'))

    vis_q.create_window("Query {}/93".format(index+1), 800, 600)
    vis_c.create_window("Correct {}/93".format(index+1), 800, 600)
    vis_w.create_window("Wrong {}/93".format(index+1), 800, 600)

    opt_q = vis_q.get_render_option()
    opt_q.background_color = np.asarray([0, 0, 0])
    opt_q.point_size = 1
    opt_q.show_coordinate_frame = False

    opt_c = vis_c.get_render_option()
    opt_c.background_color = np.asarray([0, 0, 0])
    opt_c.point_size = 1
    opt_c.show_coordinate_frame = False

    opt_w = vis_w.get_render_option()
    opt_w.background_color = np.asarray([0, 0, 0])
    opt_w.point_size = 1
    opt_w.show_coordinate_frame = False

    vis_q.add_geometry(pcd_q)
    vis_c.add_geometry(pcd_c)
    vis_w.add_geometry(pcd_w)

    vis_q.add_geometry(axis)
    vis_c.add_geometry(axis)
    vis_w.add_geometry(axis)

    # 创建 Thread 实例
    t_q = Thread(target=vis_q.run())
    t_c = Thread(target=vis_c.run())
    t_w = Thread(target=vis_w.run())

    # 启动线程运行
    t_q.start()
    t_c.start()
    t_w.start()

    # 等待所有线程执行完毕
    t_q.join()  # join() 等待线程终止，要不然一直挂起
    t_c.join()
    t_w.join()

    # vis_q.run()
    # vis_c.run()
    # vis_w.run()


if __name__ == "__main__":
    pred_result = "../../loop/np"
    pred_name = 'pred_result_AS_(1_1500).npy'
    loop_name = 'pair_gt.csv'
    pcd_path = "/home/alex/Dataset/Apollo/SanJoseDowntown_TrainData/pcds"

    # n, q, p = load_check_file(pred_result, pred_name)
    # n = np.array(n)
    result = load_check_file_np(pred_result, pred_name)
    recall_bar(result[:, 0])

    # pair_gt = load_loop_gt(pred_result, loop_name)
    w_q, w_p = wrong_filter(pred_result, pred_name)
    # cpt = correct_pair(w_q, w_p, pair_gt)

    # print("\nAnalysis Trajectory Generating...")
    # with tqdm(total=len(cpt)) as t:
    #     for index in range(len(cpt)):
    #         T = TrajGen(cpt[index])
    #         T.layout_plot()
    #         t.update(1)
    #     t.close()

    # v_index = input("Please input the query index:")
    # v_index = int(v_index)-1
    # v_index = 82
    # vis_cpt(pcd_path, cpt, index)
