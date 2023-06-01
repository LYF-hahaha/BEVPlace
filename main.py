from __future__ import print_function
import argparse
from os.path import join, isfile
from os import environ
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import faiss
from network.bevplace import BEVPlace
from tqdm import tqdm

import dataset

# 指定所使用的GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 运行参数
parser = argparse.ArgumentParser(description='BEVPlace')
parser.add_argument('--test_batch_size', type=int, default=128, help='Batch size for testing')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=40, help='Number of threads for each data loader to use')
parser.add_argument('--resume', type=str, default='checkpoints', help='Path to load checkpoint from, for resuming training or testing.')

# 评价用的函数
def evaluate(eval_set, model):
    test_data_loader = DataLoader(dataset=eval_set, 
                                  num_workers=opt.threads,
                                  batch_size=opt.test_batch_size,
                                  shuffle=False,
                                  pin_memory=cuda)

    # 模型换为评价模式
    model.eval()

    global_features = []

    # torch.no_grad() 是 torch 中一个上下文管理器
    # 在这个上下文中，所有操作都不会被追踪以用于求导。以节省内存、加速计算。
    with torch.no_grad():
        print('====> Extracting Features')

        # 进度条展示器
        with tqdm(total=len(test_data_loader)) as t:

            # enumerate是python自带函数，用于遍历列表元素（在字典、列表上枚举）
            # 对于一个可迭代/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
            # 一般第一个接收索引号，第二个接收值
            # enumerate还可以接收第二个参数，用于指定索引起始值（此处起始值为1，默认是0）
            for iteration, (input, indices) in enumerate(test_data_loader, 1):

                batch_feature = bevplace(input)

                # detach(): 阻断反向传播(输出仍留在显存中)
                # cpu(): 输出移至cpu上
                # numpy(): tensor转numpy（方便后续存储）
                global_features.append(batch_feature.detach().cpu().numpy())
                # 手动更新进度条，在目前进度条上增加1个进度
                t.update(1)

    # vstack 把global_features竖直堆叠形成矩阵（一行代表一个特征）
    global_features = np.vstack(global_features)

    # 用事先定义好的eval_set挑出相应序号的feature
    query_feat = global_features[eval_set.num_db:].astype('float32')
    db_feat = global_features[:eval_set.num_db].astype('float32')

    print('====> Building faiss index')
    # 一种索引方法
    faiss_index = faiss.IndexFlatL2(query_feat.shape[1])
    faiss_index.add(db_feat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(query_feat, max(n_values)) 

    gt = eval_set.getPositives() 

    correct_at_n = np.zeros(len(n_values))
    whole_test_size = 0

    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) ==0 : 
            continue
        whole_test_size+=1
        for i,n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / whole_test_size

    recalls = {} 
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    return recalls


if __name__ == "__main__":
    opt = parser.parse_args()

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
    # 载入数据集
    print('===> Loading dataset(s)')
    data_path = './data/KITTI05/'
    seq = '05'
    # 点云&bev_image对应（seq在此无特别作用，应该是作者在训练全部21个序列时选则序列用的）
    eval_set = dataset.KITTIDataset(data_path, seq)

    # 载入模型（只是 恢复ckpt的目的是什么？）
    print('===> Building model')
    bevplace = BEVPlace()
    resume_ckpt = join(opt.resume, 'checkpoint.pth.tar')
    print("=> loading checkpoint '{}'".format(resume_ckpt))

    # 加载保存的模型（ torch.save() ）
    # 官方文档：Load all tensors onto the CPU, using a function
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    
    # PyTorch 中模型加载权重的一种方法。
    # 它需要一个字典作为参数，其中包含了模型的权重和其他参数。
    # 这个字典可以使用 torch.save() 函数保存到磁盘上，并使用 torch.load() 函数读取。
    # 使用这种方法加载模型时，模型的结构必须与保存时的结构完全相同。
    bevplace.load_state_dict(checkpoint['state_dict'])

    # 模型放到GPU上去
    bevplace = bevplace.to(device)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_ckpt, checkpoint['epoch']))

    # 多GPU并行（要总共就一个GPU，那就不说了）
    bevplace = nn.DataParallel(bevplace)
    model = bevplace.to(device)

    # 返回recall
    recalls = evaluate(eval_set, model)
    