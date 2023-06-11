import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from sklearn.neighbors import NearestNeighbors
from network.utils import TransformerCV
from network.groupnet import group_config


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
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
        query_frames = {'SanJose_train':range(0, 4000),
                        'Baylands_train':range(2500, 3500),
                        'ColumbiaPark_02':range(3500, 10000),
                        'Sunnyvale_Caspian':range(0, 4000)}
        db_frames = {'SanJose_train':range(4001, 16596),
                     'Baylands_train':range(3501, 6400),
                     'ColumbiaPark_02':range(10001, 25000),
                     'Sunnyvale_Caspian':range(4001, 14000)}


        self.pos_threshold = 2  # ground truth threshold

        # preprocessor
        self.input_transform = input_transform()
        self.transformer = TransformerCV(group_config)
        self.pts_step = 5

        # root pathes
        bev_path = data_path + seq + '/imgs/'
        # lidar_path = data_path + '/velodyne/'

        # geometry positions
        poses = np.loadtxt(data_path + seq + '/gt_poses.txt')
        # 前3列和最后一列合一起？得看pose.txt文件定义了
        positions = np.hstack([poses[:, 2].reshape(-1, 1), poses[:, 3].reshape(-1, 1)])

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
        xs, ys = np.meshgrid(np.arange(self.pts_step, img.size()[1] - self.pts_step, self.pts_step),
                             np.arange(self.pts_step, img.size()[2] - self.pts_step, self.pts_step))
        xs = xs.reshape(-1, 1)
        ys = ys.reshape(-1, 1)
        pts = np.hstack((xs, ys))
        img = img.permute(1, 2, 0).detach().numpy()
        transformed_imgs = self.transformer.transform(img, pts)
        data = self.transformer.postprocess_transformed_imgs(transformed_imgs)
        return data

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.input_transform(img)
        img *= 255
        img = self.transformImg(img)

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