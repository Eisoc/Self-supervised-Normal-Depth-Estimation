import os
from typing import *
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image


class KITTIMotion(Dataset):
    def __init__(self, root_dir: str, train: bool) -> None:
        super().__init__()
        self.train = train
        self.train_data_path = os.path.join(root_dir, 'train')
        self.val_data_path = os.path.join(root_dir, 'val')

        self.train_image = []
        self.train_flow = []
        self.train_label = []
        self.val_image = []
        self.val_flow = []
        self.val_label = []

        if self.train:
            image = glob.glob(os.path.join(self.train_data_path, 'image/*.png'))
            flow = glob.glob(os.path.join(self.train_data_path, 'flow/*.png'))
            label = glob.glob(os.path.join(self.train_data_path, 'label/*.png'))
            self.train_data_len = len(label)
        else:
            image = glob.glob(os.path.join(self.val_data_path, 'image/*.png'))
            flow = glob.glob(os.path.join(self.val_data_path, 'flow/*.png'))
            label = glob.glob(os.path.join(self.val_data_path, 'label/*.png'))
            self.val_data_len = len(label)
        self.process(image, flow, label)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # rotate = iaa.Affine(rotate=(-45, 45), cval=2)
        # img, flow, label = self.train_image[index].numpy().astype(np.uint8), self.train_flow[index].numpy().astype(np.uint8), self.train_label[index].numpy().astype(np.uint8)
        # img, flow, label = rotate(images=[img, flow, label])
        if self.train:
            return (self.train_image[index], self.train_flow[index], self.train_label[index])
        else:
            return (self.val_image[index], self.val_flow[index], self.val_label[index])
        
    def __len__(self) -> int:
        if self.train:
            return self.train_data_len
        else:
            return self.val_data_len

    def process(self, image, flow, label):
        for i, f, l in zip(image, flow, label):
            i_m = Image.open(i)
            f_m = Image.open(f)
            l_m = Image.open(l)
            l_m = np.array(l_m)[:, :, 0] / 100
            l_m = l_m.astype(np.int64)
            trans = transforms.ToTensor()
            if self.train:
                self.train_image.append(trans(i_m))
                self.train_flow.append(trans(f_m))
                self.train_label.append(torch.from_numpy(l_m).long())
            else:
                self.val_image.append(trans(i_m))
                self.val_flow.append(trans(f_m))
                self.val_label.append(torch.from_numpy(l_m).long())
            

if __name__ == '__main__':
    data = KITTIMotion('data', True)
    print(len(data))
    print(data[0])