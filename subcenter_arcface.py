#!/usr/bin/env python
# encoding: utf-8

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class Subcenter_Arcface(nn.Module):
    def __init__(self, in_feature=512, out_feature=10575, k = 4,s=32.0, m=0.50, easy_margin=False):
        super(Subcenter_Arcface, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.k = k
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature * k, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m


    def forward(self, x, label):
        # cos(theta)
        cosine_all = F.linear(F.normalize(x), F.normalize(self.weight))
		if self.k > 1:
        	cosine = cosinel.view(-1, self.out_feature, self.k)
        	cosine = torch.max(cosine, 2)[0]   #这里论文用的max pooling，mxnet远吗用的“mx.symbol.max(sim_s3, axis=2)”

        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output



if __name__ == '__main__':
    # pass

    input = torch.randn(10,512)
    label = torch.LongTensor(10).random_(9)
    head = Subcenter_Arcface()
    out = head(input,label)
    print(out.shape)
