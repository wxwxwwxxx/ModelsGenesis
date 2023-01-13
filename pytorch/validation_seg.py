#!/usr/bin/env python
# coding: utf-8
# Segmentation

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import sys
from utils import *
import unet3d
from config import seg_config, lcs_config,bms_config
from tqdm import tqdm
import lmdbdataset
from dataloader import ncs_dataset,lcs_dataset,bms_dataset
from torch.utils.data import DataLoader
import yaml
from torch.utils.tensorboard import SummaryWriter
import logging
import custom_transform
from torchvision import transforms
import time
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# conf = seg_config(suffix="ncs_vanilla_aug_1",
#                   pretrain_weight="/ckpt/pretrain/Models/Unet3D-bs96_gradclip_fixed/Genesis_Chest_CT.pt")
conf = lcs_config(suffix="lcs_vanilla_fullimage_stable_alteraug_enhance_lr1e-3_12812816_1",pretrain_weight="/ckpt/pretrain/Models/Unet3D-bs96_gradclip_fixed/Genesis_Chest_CT.pt")
# conf = bms_config(suffix="bms_pretrain_stable_fullimage_12012016_1",pretrain_weight="/ckpt/pretrain/Models/Unet3D-bs96_gradclip_fixed/Genesis_Chest_CT_BMS.pt")
conf.display()

imgshow = lambda img: np.concatenate([np.concatenate([img[0, 0, :, :, z] for z in range(i, i + 8)], 0) for i in range(0, 32, 8)],
                               1)
######NCS
# num_dict = {"train":4082,"valid":3126,"test":852}
# with open(conf.split_yaml, "r", encoding="utf-8") as f:
#     num_dict = yaml.safe_load(f)
#
# train_list = []
# valid_list = []
# test_list = []
# for i in range(num_dict["train"]):
#     train_list.append(f"train_{i}")
# for i in range(num_dict["valid"]):
#     valid_list.append(f"valid_{i}")
# for i in range(num_dict["test"]):
#     test_list.append(f"test_{i}")
#
# test_dataset = ncs_dataset(conf.data, conf.shape, conf, key_list=test_list)
# test_dataloader = DataLoader(test_dataset, 1)

######LCS
test_dataset = lcs_dataset(conf.data, conf.shape, conf, key_index_list=conf.test_idx)
test_dataloader = DataLoader(test_dataset, 1)



######BMS
# t0 = transforms.Compose([
#     custom_transform.ZoomScale3D(0.5),
#     custom_transform.CutBlack3D(0.0),
#     custom_transform.FixedAlign3D(120,120,80)
# ])


# test_dataset = bms_dataset(conf.data, conf.shape, conf, key_list=conf.val_key_list, t=t0)
# test_dataloader = DataLoader(test_dataset, 1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet3d.UNet3D()
# model = unet3d.UNet3D_bms()
model.to(device)

# testing

checkpoint = torch.load(os.path.join(conf.model_path, "lcs_downstream.pt"))
model.load_state_dict(checkpoint['state_dict'])
print(f"Loading weights from {os.path.join(conf.model_path, 'lcs_downstream.pt')}")
print(len(test_dataloader))
count = 0
with torch.no_grad():
    model.eval()
    print("testing....")
    test_ious = []
    test_dices = []
    test_mious = []
    test_img_gt = []
    test_img_pred = []


    for x, y in test_dataloader:
        print(count,end=' ')
        count = count + 1
        image = x.float().to(device)
        gt = y.float().to(device)
        pred = model(image)


        iou_metric = iou(gt, pred)
        dice_metric = dice_coef(gt, pred)
        miou_metric = binary_mean_iou_eval(gt, pred)

        if not math.isnan(iou_metric.item()):

            test_ious.append(iou_metric.item())
        test_dices.append(dice_metric.item())
        test_mious.append(miou_metric.item())
        # img_cpu = x.cpu().numpy()
        # gt_cpu = y.cpu().numpy()
        # pred_cpu = pred.cpu().numpy()


        # pred_full_cpu = torch.tensor(pred_full_cpu[None,None,:,:,:,0])
        # test_full_mious.append(binary_mean_iou_eval(gt_full, pred_full_cpu).item())



# logging

test_iou = np.average(test_ious)
test_dice = np.average(test_dices)
miou = np.average(test_mious)


print("")
print(f"Test IOU = {test_iou}")
print(f"Test Dice = {test_dice}")
print(f"Test M-IOU = {miou}")

