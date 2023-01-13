#!/usr/bin/env python
# coding: utf-8


import warnings

warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *
import unet3d
from config import models_genesis_config
from tqdm import tqdm
import lmdbdataset
from dataloader import lmdb_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

print("torch = {}".format(torch.__version__))

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Config
conf = models_genesis_config()
conf.display()
writer = SummaryWriter(conf.tboard_path)

# Dataset
split_dict = lmdbdataset.lmdbdataset.load_train_val_list(conf.split_yaml)

train_dataset = lmdb_dataset(conf.data, conf.shape, conf, key_list=split_dict["train"])
train_dataloader = DataLoader(train_dataset, conf.batch_size, True, num_workers=conf.workers, drop_last=False,persistent_workers=True)

val_dataset = lmdb_dataset(conf.data, conf.shape, conf, key_list=split_dict["val"])
val_dataloader = DataLoader(val_dataset, 4, shuffle=True, num_workers=2,persistent_workers=True,drop_last=True)

# Model Init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet3d.UNet3D()
model.to(device)
print("Train Data Amount:", len(train_dataset))
print("Test Data Amount:", len(val_dataset))
print("Total CUDA devices: ", torch.cuda.device_count())

summary(model, (1, conf.input_rows, conf.input_cols, conf.input_deps), batch_size=-1)
criterion = nn.MSELoss()

if conf.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
    raise NotImplementedError


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True,
                                                       min_lr=1e-6, eps=1e-3)

# Metrics Init
train_losses = []
valid_losses = []

best_loss = 100000
intial_epoch = 0
num_epoch_no_improvement = 0
sys.stdout.flush()
global_iteration = 0
# For Resume
if conf.weights != None:
    checkpoint = torch.load(conf.weights)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    intial_epoch = checkpoint['epoch']
    global_iteration = checkpoint['global_iter']
    best_loss = checkpoint['best_loss']
    print("Loading weights from ", conf.weights)
else:
    print("Training from scratch...")
sys.stdout.flush()

for epoch in range(intial_epoch, conf.nb_epoch):

    model.train()
    iteration = 0
    for image, gt in train_dataloader:
        image, gt = image.float().to(device), gt.float().to(device)
        pred = model(image)
        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        # clip grad
        for n, p in model.named_parameters():
            if torch.any(torch.isnan(p.grad)).item():
                print("NaN grad detected.")
            p.grad.data = torch.where(torch.isnan(p.grad.data), torch.full_like(p.grad.data, 0), p.grad.data)
        optimizer.step()
        train_losses.append(round(loss.item(), 2))
        writer.add_scalar('Loss/train_iter', loss.item(), global_iteration)
        #TODO: add interval to config
        if iteration % 100 == 0:
            print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                  .format(epoch + 1, conf.nb_epoch, iteration, np.average(train_losses[-100:])))#-100 <-- interval
            writer.add_images('Image/train_pred', pred[0:16, :, :, :, 16], global_iteration)
            writer.add_images('Image/train_gt', gt[0:16, :, :, :, 16], global_iteration)
            writer.add_images('Image/train_input', image[0:16, :, :, :, 16], global_iteration)
            sys.stdout.flush()
        iteration += 1
        global_iteration += 1
    with torch.no_grad():
        model.eval()
        print("validating....")
        valid_pred_list = []
        valid_gt_list = []
        valid_input_list = []
        for x, y in val_dataloader:
            image = x.to(device).float()
            gt = y.to(device).float()
            pred = model(image)
            #TODO: 16 can be added into config
            if len(valid_gt_list) < 16:
                valid_pred_list.append(pred)
                valid_gt_list.append(gt)
                valid_input_list.append(image)
            loss = criterion(pred, gt)
            valid_losses.append(loss.item())

    # logging
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    valid_pred = torch.cat(valid_pred_list, 0)
    valid_gt = torch.cat(valid_gt_list, 0)
    valid_input = torch.cat(valid_input_list, 0)

    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss, train_loss))
    writer.add_scalar('Loss/train_epoch', train_loss, epoch)
    writer.add_scalar('Loss/val_epoch', valid_loss, epoch)
    writer.add_images('Image/valid_pred', valid_pred[0:16, :, :, :, 16], epoch)
    writer.add_images('Image/valid_gt', valid_gt[0:16, :, :, :, 16], epoch)
    writer.add_images('Image/valid_input', valid_input[0:16, :, :, :, 16], epoch)

    train_losses = []
    valid_losses = []
    scheduler.step(valid_loss)
    if valid_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        # save model
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_iter': global_iteration + 1,
            'best_loss' : best_loss
        }, os.path.join(conf.model_path, "Genesis_Chest_CT.pt"))
        print("Saving model ", os.path.join(conf.model_path, "Genesis_Chest_CT.pt"))
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                  num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == conf.patience:
        print("Early Stopping")
        break
    sys.stdout.flush()
