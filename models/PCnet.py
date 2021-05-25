import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch import Tensor
import numpy as np

from .model_utils import *



class PCnet(nn.Module):

    def __init__(self,
                 n_neighbor=20,
                 num_classes=40,
                 class_weight=None,
                 gamma=9.6,
                 beta=0.83,
                 lamda=10):
        super(PCnet, self).__init__()

        self.n_neighbor = n_neighbor
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d21 = conv_2d(64, 128, 1)
        self.conv2d22 = conv_2d(64, 128, 1)
        self.conv2d31 = conv_2d(64, 128, 1)
        self.conv2d32 = conv_2d(128, 128, 1)
        self.conv2d41 = conv_2d(128, 128, 1)
        self.conv2d42 = conv_2d(64, 128, 1)

        self.mlp1 = nn.Sequential(fc_layer(384, 512, bn=1, relu=1),
                                  nn.Dropout(p=0.5))
        self.mlp2 = nn.Sequential(fc_layer(512, 256, bn=1, relu=1),
                                  nn.Dropout(p=0.5))

        self.mlp3 = AdaptiveMarginLoss(emdsize=256,
                                       class_num=num_classes,
                                       class_weight=class_weight,
                                       gamma=gamma,
                                       beta=beta,
                                       lamda = lamda)

    def forward(self,
                x,
                label=None,
                get_fea=False,
                is_normal=False,
                test=False):

        x = x.permute(0, 2, 1).unsqueeze(-1).float()
        x_edge = get_graph_feature(x, self.n_neighbor).float()
        x_trans = self.trans_net(x_edge)
        x = x.squeeze(-1).transpose(2, 1)
        x = torch.bmm(x, x_trans)
        x = x.transpose(2, 1)

        x1 = get_graph_feature(x, self.n_neighbor)
        x1 = self.conv2d1(x1)
        x1, _ = torch.max(x1, dim=-1, keepdim=True)

        x2 = get_graph_feature(x1, self.n_neighbor)
        x2 = self.conv2d2(x2)
        x2, _ = torch.max(x2, dim=-1, keepdim=True)

        x3 = get_graph_feature(x2, self.n_neighbor)
        x3 = self.conv2d3(x3)
        x3, _ = torch.max(x3, dim=-1, keepdim=True)

        x4 = get_graph_feature(x3, self.n_neighbor)
        x4 = self.conv2d4(x4)
        x4, _ = torch.max(x4, dim=-1, keepdim=True)

        x231 = self.conv2d21(x2)
        x232 = self.conv2d22(x3)
        x23 = x231 * x232
        x23 = torch.sum(x23, dim=-2).reshape(x23.shape[0], -1)
        x23 = torch.sqrt(x23 + 1e-5)
        x23 = torch.nn.functional.normalize(x23)

        x341 = self.conv2d31(x3)
        x342 = self.conv2d32(x4)
        x34 = x341 * x342
        x34 = torch.sum(x34, dim=-2).reshape(x34.shape[0], -1)
        x34 = torch.sqrt(x34 + 1e-5)
        x34 = torch.nn.functional.normalize(x34)

        x141 = self.conv2d41(x4)
        x142 = self.conv2d42(x1)
        x14 = x141 * x142
        x14 = torch.sum(x14, dim=-2).reshape(x14.shape[0], -1)
        x14 = torch.sqrt(x14 + 1e-5)
        x14 = torch.nn.functional.normalize(x14)
        x5 = torch.cat((x23, x34, x14), dim=1)

        out1 = self.mlp1(x5)
        fea = self.mlp2(out1)

        out, loss = self.mlp3(fea, label, test)

        if get_fea:
            if is_normal:
                fea = fea / torch.norm(fea, 2, 1, True)
            return out, fea, loss
        else:
            return out, loss


class AdaptiveMarginLoss(nn.Module):

    def __init__(self,
                 emdsize,
                 class_num,
                 class_weight=None,
                 gamma=9.6,
                 beta=0.83,
                 lamda=10):

        super(AdaptiveMarginLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.class_num = class_num
        self.emdsize = emdsize
        self.gamma = gamma
        self.beta = beta
        self.lamda = lamda

        self.weight = nn.Parameter(
            torch.FloatTensor(self.class_num, self.emdsize))
        nn.init.xavier_uniform_(self.weight)

        self.class_weight = class_weight

    def forward(self, input, label=None, test=False) -> Tensor:

        batch_size = input.size()[0]
        input = nn.functional.normalize(input, p=2, dim=1, eps=1e-12)
        weight = nn.functional.normalize(self.weight, p=2, dim=1, eps=1e-12)
        fea = nn.functional.linear(input, weight)

        if test:
            return fea, torch.zeros((1))

        one_hot = torch.zeros(fea.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = one_hot.type(dtype=torch.bool)
        sp = fea[one_hot].view(batch_size, -1)
        sn = fea[one_hot.logical_not()].view(batch_size, -1)

        margin = self.class_weight.repeat(batch_size, 1)
        m_j = margin[one_hot.logical_not()].view(batch_size, -1)

        logit = sn - sp + self.beta * m_j.to(sp.device)

        logit = torch.logsumexp(self.gamma*logit, dim=1)
        loss = self.soft_plus(logit)

        score = torch.mm(weight, weight.t())
        score = torch.clamp(score, min=-1, max=1)
        score = score - torch.diag(torch.diag(score, 0))
        score = score[label]
        score = torch.max(score, 1)[0]
        loss_w = self.soft_plus(torch.mean(score) + 0.5)
        loss += self.lamda*loss_w

        return fea, loss.mean()



