#!/usr/bin/env python
# coding: utf-8
# Segmentation

import warnings

warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *
import unet3d
from config import seg_config
from tqdm import tqdm
import lmdbdataset
from dataloader import ncs_dataset
from torch.utils.data import DataLoader
import yaml
from torch.utils.tensorboard import SummaryWriter
import logging
import custom_transform
from torchvision import transforms



os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# conf = seg_config(suffix="ncs_vanilla_alteraug_2",pretrain_weight=None)
conf = seg_config(suffix="ncs_pretrain_alteraug_new_2",pretrain_weight="/ckpt/pretrain/Models/Unet3D-bs96_gradclip_fixed/Genesis_Chest_CT.pt")
conf.display()
writer = SummaryWriter(conf.tboard_path)

### logger

logger = logging.getLogger("Downstream")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(conf.logs_path, "output.log"))
fh.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(asctime)s,%(message)s', datefmt='%H:%M:%S')
fh_formatter = logging.Formatter('%(asctime)s,[%(name)s],%(message)s', datefmt='%Y/%m/%d %H:%M:%S')
ch.setFormatter(ch_formatter)
fh.setFormatter(fh_formatter)
logger.addHandler(ch)
logger.addHandler(fh)



logger.info("torch = {}".format(torch.__version__))

# num_dict = {"train":4082,"valid":3126,"test":852}
with open(conf.split_yaml,"r",encoding="utf-8") as f:
    num_dict = yaml.safe_load(f)

train_list = []
valid_list = []
test_list = []
for i in range(num_dict["train"]):
    train_list.append(f"train_{i}")
for i in range(num_dict["valid"]):
    valid_list.append(f"valid_{i}")
for i in range(num_dict["test"]):
    test_list.append(f"test_{i}")
t = transforms.Compose([

    custom_transform.RandomTransposingXY(0.5),
    custom_transform.RandomFlipping(0.5),
    transforms.RandomApply([custom_transform.RandomNoiseXY(0.01,seg=True)], p=0.5),
    transforms.RandomApply([custom_transform.RandomRotationXY(180)], p=0.5),
    custom_transform.RandomAlign3D(64, 64, 32)
])
train_dataset = ncs_dataset(conf.data, conf.shape, conf, key_list=train_list,augment=t)
train_dataloader = DataLoader(train_dataset, conf.batch_size, True, num_workers=conf.workers, drop_last=False, persistent_workers=True)


val_dataset = ncs_dataset(conf.data, conf.shape, conf, key_list=valid_list)
val_dataloader = DataLoader(val_dataset, 1)


test_dataset = ncs_dataset(conf.data, conf.shape, conf, key_list=test_list)
test_dataloader = DataLoader(test_dataset, 1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet3d.UNet3D()
model.to(device)

logger.info(f"Total CUDA devices: {torch.cuda.device_count()}")

summary(model, (1, conf.input_rows, conf.input_cols, conf.input_deps), batch_size=-1)
criterion = dice_coef_loss


# logger.info("Freeze encoder")
# for n,t in model.named_parameters():
#     if 'down' in n:
#         t.requires_grad = False

if conf.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
    raise NotImplementedError

# Change to
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True,
                                                       min_lr=1e-6, eps=1e-4)


best_loss = 0.0
intial_epoch = 0
num_epoch_no_improvement = 0

# pretrain
if conf.weights != None:
    checkpoint = torch.load(conf.weights)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #intial_epoch = checkpoint['epoch']
    #best_loss = checkpoint['best_loss']
    logger.info(f"Loading weights from {conf.weights}" )
else:
    logger.info("Training from scratch...")
#resume is not need

for epoch in range(intial_epoch, conf.nb_epoch):
    # scheduler.step(epoch)
    model.train()
    iteration = 0
    train_losses = []

    for image, gt in train_dataloader:
        image, gt = image.to(device).float(), gt.to(device).float()
        pred = model(image)
        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))

        if (iteration + 1) % 50 == 0:
            logger.debug('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                  .format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses[-50:])))
            writer.add_images('Image/train_pred', pred[0:16, :, :, :, 16], epoch)
            writer.add_images('Image/train_gt', gt[0:16, :, :, :, 16], epoch)
            writer.add_images('Image/train_input', image[0:16, :, :, :, 16], epoch)
        iteration += 1



    with torch.no_grad():
        model.eval()
        logger.info("validating....")
        valid_pred_list = []
        valid_gt_list = []
        valid_input_list = []
        valid_losses = []
        valid_ious = []
        valid_dices = []
        for x, y in val_dataloader:

            image = x.to(device).float()
            gt = y.to(device).float()
            pred = model(image)
            loss = criterion(pred, gt)
            if len(valid_gt_list) < 16:
                valid_pred_list.append(pred)
                valid_gt_list.append(gt)
                valid_input_list.append(image)
            iou_metric = binary_mean_iou_eval(gt, pred)

            dice_metric = dice_coef(gt, pred)

            valid_ious.append(iou_metric.item())
            valid_dices.append(dice_metric.item())
            valid_losses.append(loss.item())
            # logging
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    valid_iou = np.average(valid_ious)
    valid_dice = np.average(valid_dices)
    valid_pred = torch.cat(valid_pred_list, 0)
    valid_gt = torch.cat(valid_gt_list, 0)
    valid_input = torch.cat(valid_input_list, 0)
    scheduler.step(valid_loss)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid_iou', valid_iou, epoch)
    writer.add_scalar('Loss/valid_dice', valid_dice, epoch)
    writer.add_scalar('Loss/valid_loss', valid_loss, epoch)
    writer.add_images('Image/valid_pred', valid_pred[:, :, :, :, 16], epoch)
    writer.add_images('Image/valid_gt', valid_gt[:, :, :, :, 16], epoch)
    writer.add_images('Image/valid_input', valid_input[:, :, :, :, 16], epoch)

    logger.info("Epoch {}, validation iou is {:.4f}, validation dice is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_iou, valid_dice, train_loss))

    if valid_iou > best_loss:
        logger.info("Validation iou increases from {:.4f} to {:.4f}".format(best_loss, valid_iou))
        best_loss = valid_iou
        num_epoch_no_improvement = 0
        # save model
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, os.path.join(conf.model_path, "ncs_downstream.pt"))
        logger.info(f"Saving model to {os.path.join(conf.model_path, 'ncs_downstream.pt')}" )
    else:
        logger.info("Validation iou does not increase from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                  num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == conf.patience:
        logger.info("Early Stopping")
        break

#testing

checkpoint = torch.load(os.path.join(conf.model_path, "ncs_downstream.pt"))
model.load_state_dict(checkpoint['state_dict'])
logger.info(f"Loading weights from {os.path.join(conf.model_path, 'ncs_downstream.pt')}" )
with torch.no_grad():
    model.eval()
    logger.info("testing....")
    test_ious = []
    test_dices = []
    test_miou = []
    for x, y in test_dataloader:

        image = x.float().to(device)
        gt = y.float().to(device)
        pred = model(image)

        iou_metric = iou(gt, pred)
        dice_metric = dice_coef(gt, pred)
        miou_metric = binary_mean_iou_eval(gt,pred)

        test_ious.append(iou_metric.item())
        test_dices.append(dice_metric.item())
        test_miou.append(miou_metric.item())
# logging

test_iou = np.average(test_ious)
test_dice = np.average(test_dices)
logger.info(f"Test IOU = {test_iou}")
logger.info(f"Test Mean-IOU = {test_miou}")
logger.info(f"Test Dice = {test_dice}")