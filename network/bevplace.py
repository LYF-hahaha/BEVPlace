import torch.nn as nn

from network.netvlad import NetVLAD
from network.groupnet import GroupNet
from network.utils import to_cuda


# 我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法。但有一些注意技巧：
#（1）一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然我也可以吧不具有参数的层也放在里面；
#（2）一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，也可不放在构造函数中
#    如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替
#（3）forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。

# BEVPlace是继承nn.Module的
class BEVPlace(nn.Module):
    def __init__(self):
        # 找到子类的父类，把子类的self转化为父类的对象
        # 父类调用自己的init函数
        # 相当于把父类的init放到子类的init中，子类就有了父类的init内容
        super(BEVPlace, self).__init__()
        self.encoder = GroupNet()
        self.netvlad = NetVLAD()

    def forward(self, input):
        input = to_cuda(input)
        local_feature = self.encoder(input) 
        local_feature = local_feature.permute(0, 2, 1).unsqueeze(-1)
        global_feature = self.netvlad(local_feature) 

        return global_feature
