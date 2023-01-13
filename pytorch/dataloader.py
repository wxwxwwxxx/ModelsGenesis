from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
from lmdbdataset import lmdbdataset
import numpy as np
from utils import generate_pair_for_pytorch
import time
import custom_transform
from torchvision import transforms
import random

class lmdb_dataset(Dataset):
    def init(self):
        self.lmdb = lmdbdataset(self.lmdb_dir, self.shape)
        if self.key_list is None:
            self.key_list = self.lmdb.key_list()
        self.init_bool = True
    def get_key_list(self):
        # The function to be run when key list is unknown
        key_list = self.lmdb.key_list()
        # the lmdb is closed, when running this function, the dataset is still uninit
        return key_list

    def __init__(self, lmdb_dir, shape, config, key_list=None):

        if key_list is None:
            # init the lmdb temporarily
            self.lmdb = lmdbdataset(lmdb_dir, shape, lock=False)
            self.key_list = self.get_key_list()
            self.lmdb.close()
        else:
            self.key_list = key_list
        self.lmdb = None
        self.lmdb_dir = lmdb_dir
        self.shape = shape
        self.init_bool = False
        self.config = config

    def __getitem__(self, index):
        if not self.init_bool:
            self.init()
        img = self.lmdb.read_image(self.key_list[index])

        img = img[None, :, :, :]
        x, y = generate_pair_for_pytorch(img, self.config)

        return x.copy(), y.copy()

    def __len__(self):
        return len(self.key_list)

    def __del__(self):
        if self.init_bool:
            self.lmdb.close()


class ncs_dataset(lmdb_dataset):
    def __init__(self, lmdb_dir, shape, config,  key_list,augment=None ):
        super().__init__(lmdb_dir, shape, config, key_list)
        #TODO: for now, the code cannot work without hancraft keylist. to be fixed
        self.augment=augment
    def __getitem__(self, index):
        if not self.init_bool:
            self.init()

        x = self.lmdb.read_image(f"{self.key_list[index]}_img")
        y = self.lmdb.read_image(f"{self.key_list[index]}_mask")
        if self.augment is not None:
            aug_input = np.stack([x,y],axis=-1)
            aug_output = self.augment(aug_input)
            x = aug_output[..., 0]
            y = aug_output[..., 1]

        return np.expand_dims(x,0).copy(),np.expand_dims(y,0).copy()
class ncc_batchsampler(Sampler):
    def __init__(self, ncc_dataset, batch_size):

        super().__init__(ncc_dataset)
        self.pos_list, self.neg_list = ncc_dataset.get_pos_neg_list()
        self.bs = batch_size
    def __iter__(self):
        random.shuffle(self.pos_list)
        random.shuffle(self.neg_list)

        #
        for i in range(0,len(self.pos_list),self.bs):

            ret_list_pos = self.pos_list[i:i+self.bs]
            ret_list_neg = self.neg_list[i:i+self.bs]
            ret = [*ret_list_pos,*ret_list_neg]
            random.shuffle(ret)
            yield ret
    def __len__(self):
        return len(self.pos_list)//self.bs
class ncc_dataset(lmdb_dataset):
    def get_key_list(self):
        key_list_full = self.lmdb.key_list()
        key_list = []
        subset_set = set(self.subset)
        for i in key_list_full:
            subset_str = i.split('_')[-2]
            subset_index = int(subset_str[-1])
            if subset_index in subset_set:
                key_list.append(i)
        return key_list
    def get_pos_neg_list(self):
        pos_list = []
        neg_list = []
        for i,n in enumerate(self.key_list):
            label = int(n.split('_')[-1])
            if label == 1:
                pos_list.append(i)
            elif label == 0:
                neg_list.append(i)
        return pos_list,neg_list
    def __init__(self, lmdb_dir, shape, config, subset_list, augment=None):
        self.subset = subset_list
        super().__init__(lmdb_dir, shape, config, None)

        self.augment=augment
        # print(len(self.pos_list))
        # print(len(self.neg_list))
    def __getitem__(self, index):
        if not self.init_bool:
            self.init()

        x = self.lmdb.read_image(f"{self.key_list[index]}")
        y = np.eye(2)[int(self.key_list[index].split('_')[-1])]
        if self.augment is not None:

            aug_output = self.augment(x[...,None])
            x = aug_output[..., 0]
        return np.expand_dims(x,0).copy(),y
    # def __len__(self):
    #     return len(self.pos_list)
class lcs_dataset(lmdb_dataset):
    def __init__(self, lmdb_dir, shape, config,key_index_list, augment=None, ):
        self.key_index_list = key_index_list
        super().__init__(lmdb_dir, shape, config, None)
        self.augment = augment
    def __getitem__(self, index):
        if not self.init_bool:
            self.init()
        #patch9_97_img
        #
        x = self.lmdb.read_image(f"{self.key_list[index]}_img")
        y = self.lmdb.read_image(f"{self.key_list[index]}_seg")
        if self.augment is not None:
            aug_input = np.stack([x,y],axis=-1)
            aug_output = self.augment(aug_input)
            x = aug_output[..., 0]
            y = aug_output[..., 1]

        return np.expand_dims(x,0).copy(),np.expand_dims(y,0).copy()
    def get_key_list(self):
        # The function to be run when key list is unknown
        key_list = self.lmdb.key_list()
        # the lmdb is closed, when running this function, the dataset is still uninit
        list_set = set()
        for i in key_list:
            i_split = i.split("_")
            index = int(i_split[1])
            if index in self.key_index_list:
                list_set.add(f"{i_split[0]}_{i_split[1]}")

        return list(list_set)

class bms_dataset(lmdb_dataset):
    def __init__(self, lmdb_dir, shape, config,key_list, augment=None, ):
        super().__init__(lmdb_dir, shape, config,key_list)
        self.augment = augment
    def __getitem__(self, index):
        if not self.init_bool:
            self.init()
        # key_t1 = f"{test_name}_t1"
        # key_t1ce = f"{test_name}_t1ce"
        # key_t2 = f"{test_name}_t2"
        # key_flair = f"{test_name}_flair"
        # key_seg = f"{test_name}_seg"

        img_t1 = self.lmdb.read_image(f"{self.key_list[index]}_t1")
        img_t1ce = self.lmdb.read_image(f"{self.key_list[index]}_t1ce")
        img_t2 = self.lmdb.read_image(f"{self.key_list[index]}_t2")
        img_flair = self.lmdb.read_image(f"{self.key_list[index]}_flair")

        seg = self.lmdb.read_image(f"{self.key_list[index]}_seg")
        aug = np.stack([img_t1, img_t1ce, img_t2, img_flair, seg], axis=-1)
        if self.augment is not None:
            aug = self.augment(aug)
        x = aug[..., 0:4].transpose(3,0,1,2)
        y = aug[..., 4:5].transpose(3,0,1,2)

        return x,y
    # def get_key_list(self):
    #     # The function to be run when key list is unknown
    #     #key_list = self.lmdb.key_list()
    #     # the lmdb is closed, when running this function, the dataset is still uninit
    #     list_set = set()
    #     for i in key_list:
    #         i_split = i.split("_")
    #         index = int(i_split[1])
    #         if index in self.key_index_list:
    #             list_set.add(f"{i_split[0]}_{i_split[1]}")
    #
    #     return list(list_set)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    import time
    # img = np.load("/dataset/ncs_data/x_train_64x64x32.npy")
    # print(np.shape(img), np.max(img), np.min(img))

    from config import cls_config
    import torch
    import yaml
    # from custom_transform import RandomRotation3D

    conf = cls_config()
    # import csv
    t = transforms.Compose([

        custom_transform.RandomTransposingXY(0.5),
        custom_transform.RandomFlipping(0.5),
        transforms.RandomApply([custom_transform.RandomRotation3D(180)], p=0.5),
        custom_transform.CutBlack3D(0.0),
        transforms.RandomApply([custom_transform.RandomNoiseXY(0.1,seg=False)], p=0.5),
        custom_transform.RandomAlign3D(64,64,32)
    ])
    #
    dataset = ncc_dataset(conf.data, conf.shape, conf, subset_list=[0],augment=t)

    dataloader = DataLoader(dataset,batch_sampler=bc)
    for z in range(3):
        for i in dataloader:
            print(i[0].shape)
            print(i[1])
    # print(len(dataset))
    # print(dataset.key_list)
    # #     print(i)
    # dataloader = DataLoader(dataset, batch_size=4, drop_last=False)
    # for img in iter(dataloader):
    #
    #     num += 1
    #     img0 = img[0].numpy()
    #     img1 = img[1].numpy()
    #     plt.subplot(2,3,1)
    #     plt.imshow(img0[0,0,:,:,32])
    #     plt.subplot(2,3,2)
    #     plt.imshow(img0[0, 1, :, :, 32])
    #     plt.subplot(2,3,3)
    #     plt.imshow(img0[0,2,:,:,32])
    #     plt.subplot(2,3,4)
    #     plt.imshow(img0[0,3,:,:,32])
    #     plt.subplot(2,3,5)
    #     plt.imshow(img1[0,0,:,:,32])
    #     plt.show()
    #     print(f"iter={num},IMG[0],{torch.min(img[0])}~{torch.max(img[0])}, Nan Check: {torch.any(torch.isnan(img[0]))}")
    #     print(f"iter={num},IMG[1],{torch.min(img[1])}~{torch.max(img[1])}, Nan Check: {torch.any(torch.isnan(img[1]))}")
    #     print(img0.shape)
    #     # plt.subplot(1, 2, 1)
    #     # plt.imshow(img0[0, 0, :, :, 32])
    #     # plt.subplot(1, 2, 2)
    #     # plt.imshow(img1[0, 0, :, :, 32],vmin=-0.1,vmax=1.1)
    #     # plt.show()
    #     print(img1.shape)
    #     # time.sleep(1)

    # train_list = []
    # valid_list = []
    # test_list = []
    # for i in range(num_dict["train"]):
    #     train_list.append(f"train_{i}")
    # for i in range(num_dict["valid"]):
    #     valid_list.append(f"valid_{i}")
    # for i in range(num_dict["test"]):
    #     test_list.append(f"test_{i}")
    # r3d = RandomRotation3D(180)
    # t = transforms.Compose([
    #
    #     custom_transform.RandomTransposingXY(0.5),
    #     custom_transform.RandomFlipping(0.5),
    #     transforms.RandomApply([custom_transform.RandomNoiseXY(0.1)], p=0.5),
    #     transforms.RandomApply([custom_transform.RandomRotationXY(180)], p=0.5),
    #     custom_transform.RandomAlign3D(64, 64, 32)
    # ])
    # dataset = ncs_dataset(conf.data, (64, 64, 32), conf,key_list=train_list,augment=t)
    # dataset = ncs_dataset(conf.data, (64, 64, 32), conf, key_list=train_list)
    # dataloader = DataLoader(dataset, batch_size=2,drop_last=False)
    #
    # for img in iter(dataloader):
    #     time.sleep(10)
    #     num += 1
    #     img0 = img[0].numpy()
    #     img1 = img[1].numpy()
    #     # [bs,c,h,w,d]
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img0[0,0,:,:,16])
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(img1[0,0,:,:,16])
    #     plt.show()
    #     # print(f"iter={num},IMG[0],{torch.min(img[0])}~{torch.max(img[0])}, Nan Check: {torch.any(torch.isnan(img[0]))}")
    #     # print(f"iter={num},IMG[1],{torch.min(img[1])}~{torch.max(img[1])}, Nan Check: {torch.any(torch.isnan(img[1]))}")

    # t = transforms.Compose([
    #     custom_transform.RandomMask3D(20, 2, 0.5),
    #     transforms.RandomApply([
    #     custom_transform.RandomColorScale3D(0.1),
    #     custom_transform.RandomNoise3D(0.05),
    #     custom_transform.RandomRotation3D(10),
    #     custom_transform.RandomZoom3D(0.2),
    #     custom_transform.RandomShift3D(10),
    #     ], p=0.7),
    #     custom_transform.RandomAlign3D(128),
    #     custom_transform.RandomMask3D(20, 2, 0.5)
    # ])
