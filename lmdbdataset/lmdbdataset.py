import lmdb
import numpy as np
import pickle
import time
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

    def __init__(self,data_path,shape,lock=True):
        self.env = lmdb.open(data_path,map_size=1099511627776,lock=lock)
        self.shape = shape
        self._k_list = []
    def close(self):
        self.env.close()

    def insert_image_byte(self,k,v):
        txn = self.env.begin(write=True)
        txn.put(k.encode(),v)
        txn.commit()
        self._k_list = []

    def insert_image(self,k,v):
        txn = self.env.begin(write=True)
        txn.put(k.encode(), v.tobytes())
        txn.commit()
        self._k_list = []

    def read_image(self, k):
        txn = self.env.begin()
        return np.reshape(np.frombuffer(txn.get(k.encode())), self.shape)

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

# lmdbd = lmdbdataset('/dataset/lmdb/debug2')


# while(c<=100):
#     a = np.random.normal(loc=0.0, scale=1.0, size=(64,64,32)).astype('float32').tobytes()
#     lmdbd.insert_image(get_short_id(),a)
#     c+=1


