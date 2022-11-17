import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler

sys.path.append("..")


class BaseFeeder(data.Dataset):
    # initialization
    def __init__(self, folder_path, prefix="./", mean=0.0, std=1.0, mode='train'):
        self.folder_path = folder_path
        self.prefix = prefix
        self.files = []
        self.mean = mean
        self.std = std
        folder = os.path.join(prefix, folder_path, mode, "")
        print(folder)
        for file in os.listdir(folder):
            d = os.path.join(folder, file)
            if os.path.isdir(d):
                if os.path.isfile(d + "/{}_0".format(file) + ".png"):
                    self.files.append(d + "/{}_0".format(file))
                if os.path.isfile(d + "/{}_1".format(file) + ".png"):
                    self.files.append(d + "/{}_1".format(file))

    # getitem attribute
    def __getitem__(self, idx):
        img_path = self.files[idx] + ".png"
        image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).type(torch.FloatTensor)

        mask_path = self.files[idx] + "_cancer.png"
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
        mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)        
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        mask = torch.unsqueeze(mask, 0)

        return image, mask

    # collate_fn method
    # batch is list
    @staticmethod
    def collate_fn(batch):
        imgs, masks = list(zip(*batch))

        imgs = torch.stack(imgs, dim = 0)

        masks = torch.stack(masks, dim = 0)

        return imgs, masks

    def __len__(self):
        return len(self.files)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":

    feeder = BaseFeeder(r"E:\CSAWS\CSAW-S\CsawS\anonymized_dataset", prefix="")
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for batch_idx, data in enumerate(dataloader):
        print(data[0].shape)
        break