import numpy as np
import matplotlib.pyplot as plt
import pcl
import cv2
import os
import argparse
from PIL import Image
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

# KITTI
arg_1 = ["/home/alex/Dataset/KITTI/LiDAR_Original/00/velodyne", './KITTI00']

# Apollo
arg_2 = ["/home/alex/Dataset/Apollo/SanJoseDowntown_TrainData/pcds",
         '/home/alex/02_DL/02_BEVPlace/BEVPlace/data/Apollo']

parser = argparse.ArgumentParser(description='BEVPlace-Gen-BEV-Images')
parser.add_argument('--seq_path', type=str, default=arg_2[0], help='path to data')


def getBEV(all_points): #N*3
    
    all_points_pc = pcl.PointCloud()
    all_points_pc.from_array(all_points)
    f = all_points_pc.make_voxel_grid_filter()
    
    ls = 0.4
    
    f.set_leaf_size(ls, ls, ls)
    all_points_pc=f.filter()
    all_points = np.array(all_points_pc.to_list())

    x_min = -40
    y_min = -40
    x_max = 40 
    y_max = 40

    x_min_ind = int(x_min/0.4)
    x_max_ind = int(x_max/0.4)
    y_min_ind = int(y_min/0.4)
    y_max_ind = int(y_max/0.4)

    x_num = x_max_ind-x_min_ind+1
    y_num = y_max_ind-y_min_ind+1

    mat_global_image = np.zeros(( y_num,x_num), dtype=np.uint8)
          
    for i in range(all_points.shape[0]):
        x_ind = x_max_ind-int(all_points[i,1]/0.4)
        y_ind = y_max_ind-int(all_points[i,0]/0.4)
        if(x_ind>=x_num or y_ind>=y_num):
            continue
        if mat_global_image[ y_ind,x_ind]<10:
            mat_global_image[ y_ind,x_ind] += 1

    max_pixel = np.max(np.max(mat_global_image))

    mat_global_image[mat_global_image<=1] = 0  
    mat_global_image = mat_global_image*10
    
    mat_global_image[np.where(mat_global_image>255)]=255
    mat_global_image = mat_global_image/np.max(mat_global_image)*255

    return mat_global_image,x_max_ind,y_max_ind


def bev_check():
    path = '/home/alex/02_ML/01_BEVPlace/BEVPlace/data/Apollo/imgs'
    img_file = os.path.join(path, '1.png')
    img = Image.open(img_file)
    img = np.array(img)
    print(img.shape)


if __name__ == "__main__":

    args = parser.parse_args()
    bins_path = os.listdir(args.seq_path)
    bins_path.sort(key= lambda x: int(x[:-4]))

    with tqdm(total=len(bins_path)) as t:
        for i in range(len(bins_path)):
            b_p = os.path.join(args.seq_path, bins_path[i])
            # pcs = np.fromfile(b_p, dtype=np.float32).reshape(-1, 4)[:, :3]

            pcd_load = o3d.io.read_point_cloud(b_p)
            pcs = np.asarray(pcd_load.points)

            pcs = pcs[np.where(np.abs(pcs[:, 0]) < 25)[0], :]
            pcs = pcs[np.where(np.abs(pcs[:, 1]) < 25)[0], :]
            pcs = pcs[np.where(np.abs(pcs[:, 2]) < 25)[0], :]

            pcs = pcs.astype(np.float32)
            img, _, _ = getBEV(pcs)
            a = Path(os.path.join(arg_2[1], 'imgs'))
            if a.is_dir():
                cv2.imwrite(arg_2[1]+"/imgs/"+bins_path[i][:-4]+".png", img)
            else:
                print("There is no path name:{}".format(a))
            t.update(1)
        t.close()
exit()

#     bev_check()
