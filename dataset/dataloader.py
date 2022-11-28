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
from torchvision import transforms
from PIL import Image

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
    def __init__(self, folder_path, prefix="./", mean=None, std=None, mode='train', resize_h=224,
    resize_w=224, img_dim=1, data_type="folder"):
        self.folder_path = folder_path
        self.prefix = prefix
        self.files = []
        self.resize_h=resize_h
        self.resize_w=resize_w
        self.img_dim=img_dim
        self.mode = mode
        self.data_type = data_type
        folder = os.path.join(prefix, folder_path, mode, "")
        if data_type == "folder":
            for file in os.listdir(folder):
                d = os.path.join(folder, file)
                if os.path.isdir(d):
                    if os.path.isfile(d + "/{}_0".format(file) + ".png"):
                        self.files.append(d + "/{}_0".format(file))
                    if os.path.isfile(d + "/{}_1".format(file) + ".png"):
                        self.files.append(d + "/{}_1".format(file))
        else:
            for file in os.listdir(folder):
                d = os.path.join(folder, file)
                if os.path.isfile(d):
                    if os.path.isfile(d):
                        self.files.append(d)
         
        
        if mean == None or std == None:
            print("Calculating mean and std of the {} set...".format(mode))
            self.mean, self.std = self.calculate_stat()
        else:
            self.mean = mean
            self.std = std

    def calculate_stat(self):
        num_image = 0
        psum    = torch.tensor([0.0,]*self.img_dim)
        psum_sq = torch.tensor([0.0,]*self.img_dim)
        for file in self.files:
            num_image+=1

            img_path = file + ".png" if self.data_type == "folder" else file        
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            image = torch.from_numpy(cv2.resize(image, (self.resize_h,
            self.resize_w), interpolation = cv2.INTER_AREA)).type(torch.FloatTensor)
            

            psum    += image.sum()
            psum_sq += (image ** 2).sum()

        count = num_image*self.resize_h*self.resize_w                      
        total_mean = psum / count
        total_var  = (psum_sq / count) - (total_mean ** 2)
        total_std  = torch.sqrt(total_var)

        return total_mean.item(), total_std.item()


    # getitem attribute
    def __getitem__(self, idx):
        img_path = self.files[idx] + ".png" if self.data_type == "folder" else self.files[idx]       
        image = np.asarray(Image.open(img_path))
        print(image.shape)
        # resize image
        # to add data augmentation
        image = torch.from_numpy(cv2.resize(image, (self.resize_h,
        self.resize_w), interpolation = cv2.INTER_AREA)).type(torch.FloatTensor)

        # to 3d tensor
        image = torch.unsqueeze(image, 0)

        # copy and concatenate
        image  = image.repeat(3, 1, 1)
        # normalize
        image = (image - self.mean)/(self.std + 0.00001)

        
        mask_path = self.files[idx] + "_cancer.png"
        mask = np.asarray(Image.open(mask_path))
        max_px = np.amax(mask)
        if max_px > 0:
            mask = mask/max_px
        mask = cv2.resize(mask, (self.resize_w, self.resize_h), interpolation = cv2.INTER_AREA)        
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

    img = cv2.imread(r"D:\breast_cancer_classification\00002_20990909_L_CC_1.png", cv2.IMREAD_UNCHANGED)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    img2 = Image.open(r"D:\breast_cancer_classification\00002_20990909_L_CC_1.png")
    img2 = np.asarray(img2)
    print(img2.shape)
    # feeder = BaseFeeder(r"E:\CSAWS\CSAW-S\CsawS\\", prefix="",img_dim=1, data_type="folder", mean=3, std=1)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset=feeder,
    #     batch_size=2,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=0,
    # )
    # for batch_idx, data in enumerate(dataloader):
    #     print(data[0].shape)
    #     break