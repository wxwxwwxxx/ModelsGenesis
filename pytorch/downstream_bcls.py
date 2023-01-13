#!/usr/bin/env python
# coding: utf-8
# Segmentation

import warnings

warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchsummary import summary
import sys
from utils import *
import unet3d
from config import cls_config
from tqdm import tqdm
import lmdbdataset
from dataloader import ncc_dataset, ncc_batchsampler
from torch.utils.data import DataLoader
import yaml
from torch.utils.tensorboard import SummaryWriter
import logging
import custom_transform
from torchvision import transforms
from sklearn import metrics


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# conf = cls_config(suffix="ncc_pretrain_balance_sampler_3",pretrain_weight="/ckpt/pretrain/Models/Unet3D-bs96_gradclip_fixed/Genesis_Chest_CT.pt")
conf = cls_config(suffix="ncc_vanilla_balance_sampler_3",pretrain_weight=None)

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



t = transforms.Compose([

    custom_transform.RandomTransposingXY(0.5),
    custom_transform.RandomFlipping(0.5),
    transforms.RandomApply([custom_transform.RandomNoiseXY(0.1,seg=False)], p=0.5),
    transforms.RandomApply([custom_transform.RandomRotationXY(180)], p=0.5),
    custom_transform.RandomAlign3D(64, 64, 32)])

train_dataset = ncc_dataset(conf.data, conf.shape, conf, subset_list=conf.train_fold)
train_sampler = ncc_batchsampler(train_dataset, conf.batch_size)
train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=conf.workers, persistent_workers=True)


val_dataset = ncc_dataset(conf.data, conf.shape, conf,  subset_list=conf.valid_fold)
val_dataloader = DataLoader(val_dataset, 24,num_workers=2,persistent_workers=True)


test_dataset = ncc_dataset(conf.data, conf.shape, conf, subset_list=conf.test_fold)
test_dataloader = DataLoader(test_dataset, 24,num_workers=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet3d.UNet3D_bcls()
model.to(device)

logger.info(f"Total CUDA devices: {torch.cuda.device_count()}")

# summary(model, (1, conf.input_rows, conf.input_cols, conf.input_deps), batch_size=-1)
criterion = torch.nn.CrossEntropyLoss()


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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)




best_auc = 0.0
intial_epoch = 0
num_epoch_no_improvement = 0

# pretrain
if conf.weights != None:
    checkpoint = torch.load(conf.weights)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    logger.info(f"Loading weights from {conf.weights}" )
else:
    logger.info("Training from scratch...")


for epoch in range(intial_epoch, conf.nb_epoch):
    train_losses = []
    scheduler.step(epoch)
    model.train()
    iteration = 0
    for image, gt in train_dataloader:
        image, gt = image.to(device).float(), gt.to(device).float()
        pred = model(image)
        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))

        if (iteration + 1) % 100 == 0:
            logger.debug('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                  .format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses[-100:])))
            writer.add_images('Image/train_input', image[0:16, :, :, :, 16], epoch)
        iteration += 1



    with torch.no_grad():
        model.eval()
        logger.info("validating....")
        valid_losses = []
        valid_pos_score = []
        valid_gt = []
        valid_pred = []

        for x, y in val_dataloader:
            image = x.to(device).float()
            gt = y.to(device).float()
            pred_score = model(image)#[1,2]

            norm_pred_score = F.softmax(pred_score,1)#[1,2]
            pred = torch.argmax(pred_score ,dim=1)#[1]
            loss = criterion(pred_score , gt)
            valid_losses.append(loss.item())
            valid_pos_score.append(norm_pred_score[:,1])
            valid_gt.append(gt[:,1])
            valid_pred.append(pred)

    # logging
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)

    valid_pos_score = torch.cat(valid_pos_score, 0).cpu().numpy()
    valid_gt = torch.cat(valid_gt, 0).cpu().numpy()
    valid_pred = torch.cat(valid_pred, 0).cpu().numpy()
    auc = metrics.roc_auc_score(valid_gt,valid_pos_score)
    acc = metrics.accuracy_score(valid_gt,valid_pred)
    rec = metrics.recall_score(valid_gt,valid_pred)
    pre = metrics.precision_score(valid_gt,valid_pred)
    logger.info(f"Val AUC = {auc}")
    logger.info(f"Val ACC = {acc}")
    logger.info(f"Val REC = {rec}")
    logger.info(f"Val PRE = {pre}")
    writer.add_scalar('Loss/train_loss', train_loss, epoch)
    writer.add_scalar('Loss/valid_loss', valid_loss, epoch)
    writer.add_scalar('Loss/valid_auc', auc, epoch)
    writer.add_scalar('Loss/valid_acc', acc, epoch)
    writer.add_scalar('Loss/valid_pre', pre, epoch)
    writer.add_scalar('Loss/valid_rec', rec, epoch)


    if auc > best_auc:
        logger.info("Validation auc increases from {:.4f} to {:.4f}".format(best_auc, auc))
        best_auc = auc
        num_epoch_no_improvement = 0
        # save model
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_auc
        }, os.path.join(conf.model_path, "ncc_downstream.pt"))
        logger.info(f"Saving model to {os.path.join(conf.model_path, 'ncc_downstream.pt')}" )
    else:
        logger.info("Validation auc does not increases from {:.4f}, num_epoch_no_improvement {}".format(best_auc,
                                                                                                  num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == conf.patience:
        logger.info("Early Stopping")
        break

#testing

checkpoint = torch.load(os.path.join(conf.model_path, "ncc_downstream.pt"))
model.load_state_dict(checkpoint['state_dict'])
logger.info(f"Loading weights from {os.path.join(conf.model_path, 'ncc_downstream.pt')}" )
with torch.no_grad():
    model.eval()
    logger.info("testing....")
    test_pos_score = []
    test_gt = []
    test_pred = []
    for x, y in test_dataloader:

        image = x.float().to(device)
        gt = y.float().to(device)
        pred_score = model(image)
        norm_pred_score = F.softmax(pred_score, 1)  # [1,2]
        pred = torch.argmax(pred_score, dim=1)  # [1]

        loss = criterion(pred_score, gt)
        test_pos_score.append(norm_pred_score[:, 1])
        test_gt.append(gt[:, 1])
        test_pred.append(pred)

#         test_ious.append(iou_metric.item())
#         test_dices.append(dice_metric.item())
# # logging
    test_pos_score = torch.cat(test_pos_score, 0).cpu().numpy()
    test_gt = torch.cat(test_gt, 0).cpu().numpy()
    test_pred = torch.cat(test_pred, 0).cpu().numpy()
    test_auc = metrics.roc_auc_score(test_gt, test_pos_score)
    test_acc = metrics.accuracy_score(test_gt, test_pred)
    test_rec = metrics.recall_score(test_gt, test_pred)
    test_pre = metrics.precision_score(test_gt, test_pred)

    logger.info(f"Test AUC = {test_auc}")
    logger.info(f"Test ACC = {test_acc}")
    logger.info(f"Test REC = {test_rec}")
    logger.info(f"Test PRE = {test_pre}")