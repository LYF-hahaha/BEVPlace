import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import time


BASE_PATH = "/media/alex/Dataset_(SSD_2T)/benchmark_datasets_(ml3d-v2)/oxford/2014-12-10-18-10-50"
PC_STAUS = 'pointcloud_20m'
CSV_STAUS = 'pointcloud_locations_20m'

NCTL_PATH = "/media/alex/Dataset_(SSD_2T)/NCLT/2012-01-08_vel/2012-01-08/velodyne_sync"


def load_index():
    print("Loading time stamps...")
    time_stamp = []
    with open(os.path.join(BASE_PATH, CSV_STAUS+'.csv')) as index_file:
        index = index_file.read().split('\n')
        for i in range(len(index)):
            stamp = index[i].split(',')
            time_stamp.append(stamp[0])
        time_stamps = time_stamp[1:-1]
        ts_1 = sorted(time_stamps, key=float)
        return time_stamps


def load_pc_file(filename):
    #returns Nx3 matrix
    # pc = np.fromfile(os.path.join(BASE_PATH, PC_STAUS, filename+'.bin'), dtype=np.float64)
    pc = np.fromfile(os.path.join(NCTL_PATH, filename + '.bin'), dtype=np.float64)
    if pc.shape[0] != 4096*3:
        print("Error in pointcloud shape")
        return np.array([])
    pc = np.reshape(pc, (pc.shape[0]//3, 3))
    return pc


def load_pc_files():
    pcs=[]
    time_stamp = load_index()
    print("\nLoading all PC files...")
    with tqdm(total=len(time_stamp)) as t:
        for filename in time_stamp:
            #print(filename)
            pc=load_pc_file(filename)
            if pc.shape[0]!=4096:
                continue
            pcs.append(pc)
            t.update(1)
        t.close()
    pcs=np.array(pcs)
    return pcs


def vis_bin(file):
    points = load_pc_file(file)

    # 将array格式的点云转换为open3d的点云格式,若直接用open3d打开的点云数据，则不用转换
    pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
    pcd.points = o3d.utility.Vector3dVector(points) # 转换格式
    print(pcd)
    # 设置颜色 只能是0 1 如[1,0,0]代表红色为既r
    # pcd.paint_uniform_color([0, 1, 0])
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 创建窗口,设置窗口名称
    vis.create_window(window_name="bin_pc")
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为白色）
    opt.background_color = np.array([0, 0, 0])
    # 设置渲染点的大小
    opt.point_size = 5.0
    # 添加点云
    vis.add_geometry(pcd)
    vis.run()


def save_view_point():
    print("Adjust the view as you like and press \"q\" when to save the view angle")

    file_name = load_index()[0]
    pc = load_pc_file(file_name)
    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(pc)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    opt.show_coordinate_frame = False

    vis.add_geometry(pcd)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # vis.add_geometry(axis)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('viewpoint.json', param)
    vis.destroy_window()


def vis_cons():
    # files = os.listdir(files_dir)
    # files.sort(key=lambda x: int(x[:-4]))
    pcds = load_pc_files()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    opt.show_coordinate_frame = False
    # if os.path.exists("viewpoint.json"):
    #     ctr = vis.get_view_control()
    #     param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
    #     ctr.convert_from_pinhole_camera_parameters(param)
    pcd = o3d.open3d.geometry.PointCloud()

    print("\nshow result")
    with tqdm(total=len(pcds)) as t:
        for i in range(len(pcds)):
            vis.clear_geometries()
            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            pcd.points = o3d.open3d.utility.Vector3dVector(pcds[i])
            # vis.add_geometry(axis)
            vis.add_geometry(pcd)
            # ctr.convert_from_pinhole_camera_parameters(param)
            time.sleep(0.5)
            vis.run()
            t.update(1)
        t.close()
        vis.destroy_window()


if __name__ == "__main__":
    vis_bin('1326030975726043')
    # save_view_point()
    # vis_cons()
