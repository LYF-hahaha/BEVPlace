import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

def grid():
    inp = torch.ones(1, 1, 4, 4)

    # 目的是得到一个 长宽为20的tensor
    out_h = 20
    out_w = 20

    # grid的生成方式等价于用mesh_grid
    new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
    new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
    grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
    grid = grid.unsqueeze(0)

    outp = F.grid_sample(inp, grid=grid, mode='bilinear')
    print(outp.shape)  #torch.Size([1, 1, 20, 20])
    outp_2 = outp.unsqueeze(-1)
    print(outp_2.shape)


def grid_2():
    x = np.linspace(0,500,20).reshape(-1, 1)
    y = np.linspace(0,500,20).reshape(-1, 1)

    z = np.hstack((x, y))
    print(z.shape)
    # X,Y = np.meshgrid(x, y)


if __name__ == "__main__":
    grid_2()
