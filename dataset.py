import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from sklearn.neighbors import NearestNeighbors
from network.utils import TransformerCV
from network.groupnet import group_config


# 将输入图片转换成tensor张量，并归一化
def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        # 提前将1通道扩成了3通道，每个通道的初始化值不同
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])


# 导入数据，并将query和db的点云&bev_image对应好
class KITTIDataset(data.Dataset):
    def __init__(self, data_path, seq):
        super().__init__()

        #protocol setting
        db_frames = {'00': range(0, 3000), '02': range(0, 3400), '05': range(0, 1000), '06': range(0, 600)}
        query_frames = {'00': range(3200, 4541), '02': range(3600, 4661), '05': range(1200,2751), '06': range(800,1101)}
        
        self.pos_threshold = 2   #ground truth threshold
        
        #preprocessor
        self.input_transform = input_transform()
        self.transformer = TransformerCV(group_config)
        self.pts_step = 5

        #root pathes
        bev_path = data_path + '/imgs/'
        lidar_path = data_path + '/velodyne/'

        #geometry positions
        poses = np.loadtxt(data_path+'/pose.txt')
        # 12列元素 [R t]，其中第4列为z，最后一列为x （在KITTI中，z轴向前，x轴向右）
        positions = np.hstack([poses[:,3].reshape(-1,1), poses[:,11].reshape(-1,1)])

        # 取database和query，事先已经定义好了
        self.db_positions = positions[db_frames[seq], :]
        self.query_positions = positions[query_frames[seq], :]

        self.num_db = len(db_frames[seq])

        #image pathes
        # os.listdir的返回值是一个列表，列表里面存储该path下面的子目录的名称
        images = os.listdir(bev_path)
        images.sort()
        self.images = []

        for idx in db_frames[seq]:
            self.images.append(bev_path+images[idx])
        for idx in query_frames[seq]:
            self.images.append(bev_path+images[idx])     

        self.positives = None
        self.distances = None

    def transformImg(self, img):
        xs, ys = np.meshgrid(np.arange(self.pts_step,img.size()[1]-self.pts_step,self.pts_step), np.arange(self.pts_step,img.size()[2]-self.pts_step,self.pts_step))
        xs=xs.reshape(-1,1)
        ys = ys.reshape(-1,1)
        pts = np.hstack((xs,ys))
        img = img.permute(1,2,0).detach().numpy()
        transformed_imgs=self.transformer.transform(img,pts)
        data = self.transformer.postprocess_transformed_imgs(transformed_imgs)
        return data

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.input_transform(img)
        img*=255
        img = self.transformImg(img)
        
        return  img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if  self.positives is None:
            # -1代表cpu所有核都工作
            knn = NearestNeighbors(n_jobs=-1)
            # fit内表示用于训练的数据
            knn.fit(self.db_positions)

            # 直接用db的position文件计算距离，以确定正负
            self.distances, self.positives = knn.radius_neighbors(self.query_positions,
                    radius=self.pos_threshold)

        return self.positives


class YuQuanDataset(data.Dataset):
    def __init__(self, data_path='/home/luolun/gift-netvlad-kitti-test-release/data/YuQuan/'):
        super().__init__()

        #protocol setting
        db_frames = {'00': range(0,3000), '02': range(0,3400), '05': range(0,1000), '06': range(0,600)}
        query_frames = {'00': range(3200, 4541), '02': range(3600, 4661), '05': range(1200,2751), '06': range(800,1101)}
        
        self.pos_threshold = 5   #ground truth threshold
        
        #preprocessor
        self.input_transform = input_transform()
        self.transformer = TransformerCV(group_config)
        self.pts_step = 5

        #root pathes
        bev_path = data_path + '/imgs/'
        lidar_path = data_path + '/velodyne/'

        #geometry positions
        poses = np.loadtxt(data_path+'/pose.txt')
        positions = np.hstack([poses[:,4].reshape(-1,1), poses[:,8].reshape(-1,1)])

        self.db_positions = positions    #[db_frames[seq], :]
        self.query_positions = positions #[query_frames[seq], :]

        self.num_db = len(self.db_positions)

        #image pathes
        images = os.listdir(bev_path)
        images.sort()
        self.images = []
        for idx in range(len(images)):
            self.images.append(bev_path+images[idx])
        # for idx in query_frames[seq]:
        #     self.images.append(bev_path+images[idx])     

        self.positives = None
        self.distances = None

    def transformImg(self, img):
        xs, ys = np.meshgrid(np.arange(self.pts_step,img.size()[1]-self.pts_step,self.pts_step), np.arange(self.pts_step,img.size()[2]-self.pts_step,self.pts_step))
        xs=xs.reshape(-1,1)
        ys = ys.reshape(-1,1)
        pts = np.hstack((xs,ys))
        img = img.permute(1,2,0).detach().numpy()
        transformed_imgs=self.transformer.transform(img,pts)
        data = self.transformer.postprocess_transformed_imgs(transformed_imgs)
        return data

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.input_transform(img)
        img*=255
        img = self.transformImg(img)
        
        return  img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_positions)

            self.distances, self.positives = knn.radius_neighbors(self.query_positions,
                    radius=self.pos_threshold)

        return self.positives


# 导入数据，并将query和db的点云&bev_image对应好
class ApolloDataset(data.Dataset):
    def __init__(self, data_path, seq):
        super().__init__()

        # protocol setting
        query_frames = {'SanJose_train':range(10800, 11800),
                        'Baylands_train':range(2500, 3500),
                        'ColumbiaPark_02':range(3500, 10000),
                        'Sunnyvale_Caspian':range(0, 4000),
                        'train':range(1, 1500)}

        db_frames = {'SanJose_train':range(0, 10799),
                     'Baylands_train':range(3501, 6400),
                     'ColumbiaPark_02':range(10001, 25000),
                     'Sunnyvale_Caspian':range(4001, 14000),
                     'train':range(1501, 5590)}

        self.pos_threshold = 2  # ground truth threshold

        # preprocessor
        self.input_transform = input_transform()
        self.transformer = TransformerCV(group_config)

        # 为了将变换后的坐标值和变换后的图片提取出的高维特征一一对应
        # 需要事先确定一个采样点的网格
        # 这个值就是采样点网格生成时点的步长
        # f_0(g)=\phi(\eta(T_g*I), T_g(p)) 中p的生成步长
        self.pts_step = 5

        # root pathes
        bev_path = data_path + seq + '/imgs/'
        # lidar_path = data_path + '/velodyne/'

        # Apollo
        # poses = np.loadtxt(data_path + seq + '/gt_poses.txt')
        # Apollo
        # positions = np.hstack([poses[:, 2].reshape(-1, 1), poses[:, 3].reshape(-1, 1)])

        # ApolloSpace
        poses = np.loadtxt(data_path + seq + '/Overall_poses.txt')
        # ApolloSpace
        positions = np.hstack([poses[:, 1].reshape(-1, 1), poses[:, 2].reshape(-1, 1)])

        # 取database和query，事先已经定义好了
        self.db_positions = positions[db_frames[seq], :]
        self.query_positions = positions[query_frames[seq], :]

        self.num_db = len(db_frames[seq])

        # image pathes
        # os.listdir的返回值是一个列表，列表里面存储该path下面的子目录的名称
        images = os.listdir(bev_path)
        images.sort(key= lambda x: int(x[:-4]))
        self.images = []

        for idx in db_frames[seq]:
            self.images.append(bev_path + images[idx])
        for idx in query_frames[seq]:
            self.images.append(bev_path + images[idx])

        self.positives = None
        self.distances = None

    def transformImg(self, img):
        # 生成xs, ys的网格点坐标
        # xs ys都是矩阵，里面每个元素代表每个点的坐标
        xs, ys = np.meshgrid(np.arange(self.pts_step,       # X的起始点
                                       img.size()[1] - self.pts_step,    # X的终点
                                       self.pts_step),   # 采样步长
                             np.arange(self.pts_step, img.size()[2] - self.pts_step, self.pts_step)) ### Y的
        # 变成竖的
        xs = xs.reshape(-1, 1)
        ys = ys.reshape(-1, 1)
        # 左右拍一巴掌
        # 仿射变换采样点（然后拍扁了）
        # 生成网格点的坐标list，每个元素为一个点的坐标(x,y)
        pts = np.hstack((xs, ys))
        # 变成(h,w,c)
        # 不求梯度
        img = img.permute(1, 2, 0).detach().numpy()

        # 按指定的旋转和平移系数做仿射变换
        transformed_imgs = self.transformer.transform(img, pts)

        # 规范化+toTensor
        data = self.transformer.postprocess_transformed_imgs(transformed_imgs)
        # 这里面已经包含了affine后的img和pts了（以字典形式存储的，outputs=('img','pts')）
        return data

    # 当实例化对象为P后，P(index)则会运行到此处
    # 即实例化后，对实例输入某个参数便会自动进入这里（有点像针对该实例的构造函数？）
    # 返回按指定的平移和旋转系数仿射变换后的所有图片

    # 应该是需要用Dataloader，所以在这里写好了用index获取img
    # 但此时的img已包含pts
    def __getitem__(self, index):
        # 转换成RGB（其实就是扩成3通道）
        img = Image.open(self.images[index]).convert('RGB')
        # 将图片转成张量，并对3通道按不同的mean和std归一化，得出三个不同的201x201矩阵
        img = self.input_transform(img)
        # 值扩大
        img *= 255
        # 此处img已含pts
        img = self.transformImg(img)
        # img已含pts
        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_positions)

            self.distances, self.positives = knn.radius_neighbors(self.query_positions,
                    radius=self.pos_threshold)

        return self.positives
