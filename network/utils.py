import numpy as np
import torch
import os

import cv2
import torch
import pickle
import torch.nn.functional as F

from skimage.io import imread


def dim_extend(data_list):
    results = []
    for i, tensor in enumerate(data_list):
        results.append(tensor) # tensor[None,...])
    return results


def to_cuda(data):
    results = []
    for i, item in enumerate(data):
        if type(item).__name__ == "Tensor":
            results.append(item.cuda())
        elif type(item).__name__ == 'list':
            tensor_list = []
            for tensor in item:
                if type(tensor).__name__ == "Tensor":
                    tensor_list.append(tensor.cuda())
                else:
                    tensor_list2 = []
                    for tensor_i in tensor:
                        tensor_list2.append(tensor_i.cuda())
                    tensor_list.append(tensor_list2)
            results.append(tensor_list)
        else:
            raise NotImplementedError
    return results


# 插入特征
def interpolate_feats(img, pts, feats):
    # compute location on the feature map (due to pooling)
    _, _, h, w = feats.shape
    # 最后一个维度
    pool_num = img.shape[-1] // feats.shape[-1]
    pts_warp = (pts+0.5)/pool_num-0.5
    # 坐标(-1,1)化
    pts_norm = normalize_coordinates(pts_warp, h, w)
    pts_norm = torch.unsqueeze(pts_norm, 1)  # b,1,n,2

    # interpolation
    pfeats = F.grid_sample(feats, pts_norm, 'bilinear', align_corners=False)[:, :, 0, :]  # b,f,n
    pfeats = pfeats.permute(0, 2, 1) # b,n,f
    return pfeats


def l2_normalize(x, ratio=1.0, axis=1):
    norm=torch.unsqueeze(torch.clamp(torch.norm(x, 2, axis), min=1e-6), axis)
    x=x/norm*ratio
    return x


# (-1,1)化
def normalize_coordinates(coords, h, w):
    h=h-1
    w=w-1
    # 不算grad了
    coords=coords.clone().detach()
    coords[:, :, 0]-= w / 2
    coords[:, :, 1]-= h / 2

    coords[:, :, 0]/= w / 2
    coords[:, :, 1]/= h / 2
    return coords


def normalize_image(img,mask=None):
    if mask is not None: img[np.logical_not(mask.astype(np.bool))]=127
    img=(img.transpose([2,0,1]).astype(np.float32)-127.0)/128.0
    return torch.tensor(img,dtype=torch.float32)

def tensor_to_image(tensor):
    return (tensor * 128 + 127).astype(np.uint8).transpose(1,2,0)


def get_rot_m(angle):
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32) # rn+1,3,3


class TransformerCV:
    def __init__(self, config):
        ssb = config.sample_scale_begin
        ssi = config.sample_scale_inter 
        ssn = config.sample_scale_num

        srb = config.sample_rotate_begin
        sri = config.sample_rotate_inter
        srn = config.sample_rotate_num

        # 由样本缩放数、样本旋转数、样本缩放基、样本旋转基决定的缩放和旋转List
        self.scales = [ssi ** (si + ssb) for si in range(ssn)]
        self.rotations = [sri * ri + srb for ri in range(srn)]

        self.ssi=ssi

        self.ssn=ssn
        self.srn=srn

        # 缩放+旋转
        self.SRs=[]
        for scale in self.scales:
            Rs=[]
            for rotation in self.rotations:
                # 缩放乘旋转
                Rs.append(scale*get_rot_m(rotation))
            self.SRs.append(Rs)

    def transform(self, img, pts=None):
        '''
        :param img:
        :param pts:
        :return:
        '''
        h,w,_=img.shape
        # 4个顶点
        pts0=np.asarray([[0,0], [0,h], [w,h], [w,0]], np.float32)
        # 中心点位置
        center = np.mean(pts0, 0)

        # 扭曲
        pts_warps, img_warps, grid_warps = [], [], []
        img_cur=img.copy()
        # 原图缩放+旋转（析构函数里就已经干了）
        for si,Rs in enumerate(self.SRs):
            # 第一张样图不动
            if si>0:
                # 高斯模糊，缩放基不同，高斯核的值也不同
                if self.ssi<0.6:
                    img_cur=cv2.GaussianBlur(img_cur,(5,5),1.5)
                else:
                    img_cur=cv2.GaussianBlur(img_cur,(3,3),0.75)
            for M in Rs:
                # @是矩阵乘法
                # M是某个2x2的变换矩阵
                pts1 = (pts0 - center[None, :]) @ M.transpose()
                min_pts1 = np.min(pts1, 0)
                # 保留0位小数并四舍五入
                tw, th = np.round(np.max(pts1 - min_pts1[None, :], 0)).astype(np.int32)

                # compute A
                offset = - M @ center - min_pts1
                A = np.concatenate([M, offset[:, None]], 1)

                # note!!!! the border type is constant 127!!!! because in the subsequent processing, we will subtract 127
                # 仿射变换（前面计算了不同的旋转和平移系数后，在这里做仿射变换）
                img_warp=cv2.warpAffine(img_cur,A,(tw,th),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(127, 127, 127))
                img_warps.append(img_warp[:, :, :3])

                # 对采样点meshgrid也做相应的变换，并输出
                if pts is not None:
                    pts_warp = pts @ M.transpose() + offset[None, :]
                    pts_warps.append(pts_warp)
                
        outputs={'img':img_warps}
        if pts is not None: outputs['pts']=pts_warps
        return outputs

    @staticmethod
    def postprocess_transformed_imgs(results):
        img_list,pts_list,grid_list=[],[],[]
        for img_id, img in enumerate(results['img']):
            img_list.append(normalize_image(img))
            pts_list.append(torch.tensor(results['pts'][img_id], dtype=torch.float32))

        return img_list, pts_list

