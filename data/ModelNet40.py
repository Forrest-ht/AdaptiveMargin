import os
import sys
import h5py

import torch
import torch.utils.data as data
import numpy as np

from ff3d_core.operation import gather_operation, furthest_point_sample


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


class ModelNet40Cls(data.Dataset):

    def __init__(
            self,  
            root="",
            num_points=1024, 
            train=True,
            device="cuda"
    ):
        super().__init__()

        self.data_dir = os.path.join(root, "modelnet40_ply_hdf5_2048")
        self.train, self.num_points = train, num_points
        self.device = device

        if self.train:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(root, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)


    def __getitem__(self, idx):

        pt_idxs = np.arange(0, self.points.shape[1])
        if self.train:
            np.random.shuffle(pt_idxs)
        pc = self.points[idx, pt_idxs].copy()

        if self.device == "cuda":
            pc = torch.from_numpy(pc).contiguous().cuda().unsqueeze(0)
            if self.train:
                fps_idx = furthest_point_sample(pc, 1200)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(1200, self.num_points, False)]
            else:
                fps_idx = furthest_point_sample(pc, self.num_points)  # (B, npoint)
            
            pc = pc.transpose(1, 2).contiguous()
            pc = gather_operation(pc, fps_idx)
            pc = pc.transpose(1, 2).squeeze(0).contiguous().float().data.cpu().numpy()
        else:
            pc = pc[np.random.choice(2048, self.num_points, False), :]

        if self.train:
            pc = PointcloudScaleAndTranslate(pc)

        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        return pc, label.reshape(-1)

    def __len__(self):
        return self.points.shape[0]


def PointcloudScaleAndTranslate(pc, 
            scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
    xyz1 = np.random.uniform(low=scale_low, high=scale_high, size=[3])
    xyz2 = np.random.uniform(low=-translate_range, high=translate_range, size=[3])
    pc[:, 0:3] = np.multiply(pc[:, 0:3], xyz1) + xyz2
    return pc

if __name__ == "__main__":
    pass
