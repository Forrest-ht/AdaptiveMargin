import os
import sys
import os.path as osp
import numpy as np
import time
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.PCnet import PCnet
from data.ModelNet40 import ModelNet40Cls
from data.data_utils import *


use_gpu = 1
use_gpu = use_gpu and torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"


# data_root = 'path-to-ModelNet40'
data_root = '.'

train_dataset = ModelNet40Cls(num_points=1024, root=data_root, device=device)
train_loader = DataLoader(train_dataset)
test_dataset = ModelNet40Cls(num_points=1024, root=data_root, train=False, device=device)
test_loader = DataLoader(test_dataset)

num_list = [np.sum(train_dataset.labels == i) for i in range(40)]
class_weight = [
    1 - 1 / (1 + math.exp(-(i / max(num_list)))) for i in num_list
]
class_weight = [i / max(class_weight) for i in class_weight]
class_weight = torch.from_numpy(np.array(class_weight))


model = PCnet(num_classes=40, class_weight=class_weight)
# checkpoint_path = "path-to-pretrained_model"
checkpoint_path = "model_0.934_0.922.pth.tar"
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
losses = AverageMeter()
retrieval_map = RetrievalMAPMeter()

predict, label = [], []
t = 0
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        
        input = data[0].to(device)
        target = data[1].to(device).squeeze(-1)
        
        start_time = time.time()
        pred, fts, loss  = model(input, get_fea=True, is_normal=True, test=True)
        t += time.time() - start_time

        loss = loss.mean()
        losses.update(loss.data.cpu().numpy())
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred.data, -1)[1]
            
        pred, fts, target = pred.data.cpu().numpy(), fts.data.cpu().numpy(), target.data.cpu().numpy()
        retrieval_map.add(fts, target) 
        predict.append(pred) 
        label.append(target)

predict = np.concatenate(predict)
label = np.concatenate(label)
acc = np.sum(predict==label) / predict.shape[0]
mAP = retrieval_map.mAP()
t /= len(test_loader)

print(' classfication  accuracy: {:.3f}'.format(acc))  
print(' retrieval      accuracy: {:.3f}'.format(mAP))
print(' running time (ms)      : {:.1f}'.format(t*1000))
