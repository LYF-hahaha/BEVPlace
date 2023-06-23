import os
import matplotlib.pyplot as plt
import numpy as np
import re
import open3d as o3d
from tqdm import tqdm
import time
import shutil

AS_train_path = "/home/alex/Dataset/ApolloSpace/3D_detection_train/train_pose"
AS_temp_path = "/home/alex/Dataset/ApolloSpace/3D_detection_train/temp"
AS_overall_path = "/home/alex/Dataset/ApolloSpace/3D_detection_train/train_pc_overall"
AS_test_path = "/home/alex/Dataset/ApolloSpace/3D_detection_test/test_pose"


def load_AS_path():

    files = os.listdir(AS_train_path)
    files.sort(key=lambda x:int(x[7:11]))

    frame = []
    result_dict = {}
    # 文件格式为XX_num1_num2_XX
    # 将文件名按数字排序，并返回num1为key，num2为value的字典
    for i in range(len(files)):
        a = files[i].split('_')
        # 判断是否到最后一个文件夹
        if i+1 == len(files):
            frame.append(int(a[2]))
            frame.sort()
            result_dict[a[1]] = frame
            break
        frame.append(int(a[2]))
        # 判断本序号frame是否全部读完
        if a[1] != files[i+1].split('_')[1]:
            frame.sort()
            result_dict[a[1]] = frame
            frame = []

    # 将字典转变成有排序的list
    # name = "/home/alex/Dataset/ApolloSpace/3D_detection_test/sorted_test_pose_overall.txt"
    # with open(name, 'w') as f:
    #     for key in result_dict.keys():
    #         for num in result_dict[key]:
    #             f.write('result_'+key+'_'+str(num)+'_frame'+'\n')
    # f.close()
    return result_dict


def vis_traj(dict):
    result_dict = dict
    vis_dict = {}

    fig, ax = plt.subplots()
    # 按train分类
    AS_loc = []
    index_num = 1
    with open('/home/alex/Dataset/ApolloSpace/3D_detection_train/Overall_pose.txt', 'w') as oap:
        for key in result_dict.keys():
            # flag = False
            # 按frame分类
            # AS_loc = []
            for num in result_dict[key]:
                # if int(key) == 9063 and num == 10:
                    # flag = True
                loc = []
                file_name = 'result_' + key + '_' + str(num) + '_frame'
                dir = os.path.join(AS_train_path, file_name)
                pcd_files_list = os.listdir(dir)
                pcd_files_list.sort(key=lambda x:int(x[:-9]))
                for f in pcd_files_list:
                    path = os.path.join(AS_train_path, file_name, f)
                    pc_path = os.path.join(AS_temp_path, file_name, f[:-9]+'.pcd')
                    with open(path, 'r') as pose:
                        p = pose.read().split()
                        # l = [float(p[2]), float(p[3])]
                        # loc.append(l)
                        content = str(index_num) + ' ' + p[2] + ' ' + p[3] + '\n'
                        oap.write(content)
                    pose.close()
                    # shutil.copyfile(pc_path, AS_overall_path + '/' + str(index_num) + '.pcd')
                    index_num = index_num + 1
                # AS_loc.extend(loc)
    oap.close()

        # 按train分类
        # if int(key) == 9055:
        #     AS_loc = np.asarray(AS_loc)
        #     ax.scatter(AS_loc[:, 0], AS_loc[:, 1], label='test_1')
        #     AS_loc = []
        # if int(key) == 9059:
        #     AS_loc = np.asarray(AS_loc)
        #     ax.scatter(AS_loc[:, 0], AS_loc[:, 1], label='test_2')
        #     AS_loc = []
        # if flag:
        #     AS_loc = np.asarray(AS_loc)
        #     ax.scatter(AS_loc[:, 0], AS_loc[:, 1], label='test_3')
        #     AS_loc = []

        # 按frame分类
        # AS_loc = np.asarray(AS_loc)
        # ax.scatter(AS_loc[:,0], AS_loc[:,1], label = key)
    # plt.savefig('/home/alex/Dataset/ApolloSpace/3D_Decection_Tracking/pose_img/{}.png'.format(key))

    # ax.set_title('Trajectory(split_by_frame)')
    # ax.legend()
    # plt.show()


def vis_cons(files_dir):
    files = os.listdir(files_dir)
    files.sort()

    pcds = []
    print("\nloading pcd files......")
    with tqdm(total=len(files)) as t:
        for f in files:
            pcds_path = os.path.join(files_dir, f)
            x = os.listdir(pcds_path)
            x.sort()
            for p in x:
                pcd_path = os.path.join(files_dir,f,p)
                pcd = o3d.io.read_point_cloud(pcd_path)
                pcds.append(pcd)
                t.update(1)
        t.close()

    vis = o3d.visualization.Visualizer()
    print("\nshow result")
    with tqdm(total=len(pcds)) as t:
        for i in range(len(pcds)):
            vis.create_window("Progress:{:.2%}" .format(i/len(pcds)))
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            opt.point_size = 1
            opt.show_coordinate_frame = False
            vis.clear_geometries()
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            vis.add_geometry(axis)
            vis.add_geometry(pcds[i])
            # ctr.convert_from_pinhole_camera_parameters(param)
            time.sleep(0.1)
            vis.run()
            t.update(1)
        t.close()
        vis.destroy_window()


if __name__ == '__main__':
    dict = load_AS_path()
    vis_traj(dict)

    # pc_file = '/home/alex/Dataset/ApolloSpace/3D_detection_test/test_pcd_1'
    # vis_cons(pc_file)
