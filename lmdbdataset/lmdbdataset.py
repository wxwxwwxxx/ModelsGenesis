import random

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import yaml
# !/usr/bin/env python
# -*- coding:utf-8 -*-

import uuid

array = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
         "w", "x", "y", "z",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
         "W", "X", "Y", "Z"

         ]


def get_short_id():
    id = str(uuid.uuid4()).replace("-", '')  # 注意这里需要用uuid4
    buffer = []
    for i in range(0, 8):
        start = i * 4
        end = i * 4 + 4
        val = int(id[start:end], 16)
        buffer.append(array[val % 62])
    return "".join(buffer)


class lmdbdataset:

    def __init__(self,data_path,shape,lock=True, dtype="float32"):
        self.env = lmdb.open(data_path,map_size=1099511627776,lock=lock)
        self.shape = shape
        self._k_list = []
        self.dtype = dtype
    def close(self):
        self.env.close()

    def insert_image_byte(self,k,v):
        """Ensure the dtype is compatible"""
        txn = self.env.begin(write=True)
        txn.put(k.encode(),v)
        txn.commit()
        self._k_list = []

    def insert_image(self,k,v):
        txn = self.env.begin(write=True)
        txn.put(k.encode(), v.astype(self.dtype).tobytes())
        txn.commit()
        self._k_list = []

    def read_image(self, k):
        txn = self.env.begin()
        return np.reshape(np.frombuffer(txn.get(k.encode()),dtype=self.dtype), self.shape)

    def key_list(self):
        if len(self._k_list) == 0:
            i_list = []
            txn = self.env.begin()
            for key, _ in txn.cursor():
                i_list.append(key.decode())
            self._k_list = i_list

        return self._k_list

    def fn_list(self):
        k_list = self.key_list()
        f_set = set()
        for k in k_list:
            k_split = k.split('.')
            fn = ""
            for i in k_split[1:]:
                fn = fn + i + '.'
            fn = fn[:-1]
            f_set.add(fn)
        return list(f_set)

    def gen_train_val_list(self, val_ratio, save_path):
        fn_list = self.fn_list()
        patient_set = set()
        for f in fn_list:
            pid = f.split('.')[0]
            patient_set.add(pid)
        patient_val_set = set()
        for i in patient_set:
            if random.random() < val_ratio:
                patient_val_set.add(i)

        key_list = self.key_list()
        ret_dict = {"train":[],"val":[]}
        for k in key_list:
            pid = k.split('.')[1]
            if pid in patient_val_set:
                ret_dict["val"].append(k)
            else:
                ret_dict["train"].append(k)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(ret_dict, f)
    @staticmethod
    def load_train_val_list(load_path):
        with open(load_path, 'r', encoding='utf-8') as f:
            train_val_list = yaml.safe_load(f)
        return train_val_list

if __name__ == "__main__":
    pass
    # print(1)
    lmdbd = lmdbdataset('/dataset/lmdb/lits_full_scaled_128x128x16.lmdb',(128,128,16),True)
    # print(2)
    # # lmdbd.gen_train_val_list(0.01,'/dataset/lmdb/hospital_646432.yaml')
    # # l = lmdbdataset.load_train_val_list('/dataset/lmdb/hospital_646432.yaml')
    # # print(len(l["train"]))
    # # print(len(l["val"]))
    print(lmdbd.key_list())
    print(len(lmdbd.key_list()))
    index_set = set()
    for i in lmdbd.key_list():
        index_set.add(int(i.split('_')[1]))
    index_list = list(index_set)
    # print(index_list)
    # print(len(index_list))

    img1 = lmdbd.read_image('patch0_101_img')
    img2 = lmdbd.read_image('patch0_101_seg')
    # print(img1)
    # print(img2)
    plt.subplot(1,2,1)
    plt.imshow(img1[:,:,15])
    plt.subplot(1,2,2)
    plt.imshow(img2[:,:,15])
    plt.show()
    #'Brats18_2013_0_1_flair', 'Brats18_2013_0_1_seg', 'Brats18_2013_0_1_t1', 'Brats18_2013_0_1_t1ce', 'Brats18_2013_0_1_t2'
    # img1 = lmdbd.read_image('Brats18_2013_0_1_t1')
    # img2 = lmdbd.read_image('Brats18_2013_0_1_t1ce')
    # img3 = lmdbd.read_image('Brats18_2013_0_1_t2')
    # img4 = lmdbd.read_image('Brats18_2013_0_1_flair')
    # seg = lmdbd.read_image('Brats18_2013_0_1_seg')
    # plt.subplot(2,3,1)
    # plt.imshow(img1[:,:,75])
    #
    # plt.subplot(2,3,2)
    # plt.imshow(img2[:,:,75])
    # plt.subplot(2,3,3)
    # plt.imshow(img3[:,:,75])
    # plt.subplot(2,3,4)
    # plt.imshow(img4[:,:,75])
    # plt.subplot(2,3,5)
    # plt.imshow(seg[:,:,75])
    # plt.show()
    # print(img1.shape,np.min(img1),np.max(img1))
    # print(seg.shape,np.min(seg),np.max(seg))
    # print(lmdbd.key_list())
    # # print(lmdbd.fn_list())
    # print(len(lmdbd.key_list()))
    # print(len(lmdbd.fn_list()))
    # c = 0
    # fn = ""
    # while(c<=1600):
    #     a = np.random.normal(loc=0.0, scale=1.0, size=(64,64,32)).astype('float32')
    #     if c%16 ==0:
    #         fn = "EY{:0>8d}.ZX{:0>7d}.{}.nii".format(random.randint(0,99999999),random.randint(0,9999999),random.randint(0,9999))
    #     lmdbd.insert_image(f"{get_short_id()}.{fn}",a)
    #     c += 1


