import os
import open3d as o3d
import numpy as np
import time
from tqdm import tqdm


def save_view_point(pcd_path, filename):
    print("Adjust the view as you like and press \"q\" when to save the view angle")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # pcd = o3d.open3d.geometry.PointCloud()
    # pcd.points = o3d.open3d.utility.Vector3dVector(pcd_numpy)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    opt.show_coordinate_frame = False

    pcd = o3d.io.read_point_cloud(pcd_path+'/1.pcd')
    vis.add_geometry(pcd)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(axis)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def vis_cons(files_dir):
    files = os.listdir(files_dir)
    files.sort(key=lambda x: int(x[:-4]))
    pcds = []
    print("\nloading pcd files......")
    with tqdm(total=len(files)) as t:
        for f in files:
            pcd_path = os.path.join(files_dir, f)
            pcd = o3d.io.read_point_cloud(pcd_path)
            # pcd = o3d.open3d.geometry.PointCloud() # 创建点云对象
            # raw_point = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)[:, :3]
            # pcd.points= o3d.open3d.utility.Vector3dVector(raw_point) # 将点云数据转换为Open3d可以直接使用的数据类型
            # pcd = draw_color([1.0,0.36,0.2],[1.0,0.96,0.2],pcd)
            pcds.append(pcd)
            t.update(1)
        t.close()
    #if vis_detect_result:
    #    batch_results = np.load('batch_results.npy',allow_pickle=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    opt.show_coordinate_frame = False
    if os.path.exists("viewpoint.json"):
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
        ctr.convert_from_pinhole_camera_parameters(param)
    print("\nshow result")
    with tqdm(total=len(pcds)) as t:
        for i in range(len(pcds)):
            vis.clear_geometries()
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            vis.add_geometry(axis)
            vis.add_geometry(pcds[i])
            ctr.convert_from_pinhole_camera_parameters(param)
            time.sleep(0.1)
            vis.run()
            t.update(1)
        t.close()
        vis.destroy_window()


if __name__ == '__main__':
    exp_pcd_file = r"/media/alex/Dataset_(SSD_2T)/Apollo/ColumbiaPark/Apollo-SourthBay/MapData/ColumbiaPark/2018-09-21/1/pcds"
    # view_check_pcd = np.fromfile(os.path.join(exp_pcd_file,'000000.bin'), dtype=np.float32).reshape(-1, 4)[:,:3]
    save_view_point(exp_pcd_file, "viewpoint.json")
    vis_cons(exp_pcd_file)
