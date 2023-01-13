#!/usr/bin/env python
# coding: utf-8

"""
for subset in `seq 0 9`
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data /mnt/dataset/shared/zongwei/LUNA16 \
--save generated_cubes
done
"""

# In[1]:


import warnings
import datetime
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from sklearn import metrics
from optparse import OptionParser
from glob import glob
from skimage.transform import resize
import lmdbdataset
import hashlib


# assert options.data is not None
# assert options.save is not None
# assert options.fold >= 0 and options.fold <= 9
#
# if not os.path.exists(options.save):
#     os.makedirs(options.save)

class setup_config():
    hu_max = 1000.0
    hu_min = -1000.0
    HU_thred = (-150.0 - hu_min) / (hu_max - hu_min)

    def __init__(self,
                 input_rows=None,
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None,
                 crop_cols=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,
                 DATA_DIR=None,
                 len_depth=None,
                 lung_min=0.7,
                 lung_max=1.0,
                 process_num=1
                 ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.len_depth = len_depth
        self.lung_min = lung_min
        self.lung_max = lung_max
        self.process_num = process_num

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


def infinite_generator_from_one_volume(config, img_array, key_prefix, lmdb: lmdbdataset.lmdbdataset):
    size_x, size_y, size_z = img_array.shape
    if size_z - config.input_deps - config.len_depth - 1 - config.len_border_z < config.len_border_z:
        return None

    img_array[img_array < config.hu_min] = config.hu_min
    img_array[img_array > config.hu_max] = config.hu_max
    img_array = 1.0 * (img_array - config.hu_min) / (config.hu_max - config.hu_min)

    slice_set = np.zeros((config.scale, config.input_rows, config.input_cols, config.input_deps), dtype=float)

    num_pair = 0
    cnt = 0
    while True:
        cnt += 1
        if cnt > 50 * config.scale and num_pair == 0:
            return None
        elif cnt > 50 * config.scale and num_pair > 0:
            return np.array(slice_set[:num_pair])

        start_x = random.randint(0 + config.len_border, size_x - config.crop_rows - 1 - config.len_border)
        start_y = random.randint(0 + config.len_border, size_y - config.crop_cols - 1 - config.len_border)
        start_z = random.randint(0 + config.len_border_z,
                                 size_z - config.input_deps - config.len_depth - 1 - config.len_border_z)

        crop_window = img_array[start_x: start_x + config.crop_rows,
                      start_y: start_y + config.crop_cols,
                      start_z: start_z + config.input_deps + config.len_depth,
                      ]

        if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
            crop_window = resize(crop_window,
                                 (config.input_rows, config.input_cols, config.input_deps + config.len_depth),
                                 preserve_range=True,
                                 )
        crop_window = crop_window[:, :, :-1]
        t_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)
        d_img = np.ones((config.input_rows, config.input_cols, config.input_deps), dtype=float) * 2

        map0 = crop_window[:, :, 0:32] >= config.HU_thred
        t_img = np.where(map0, crop_window[:, :, 0:32], t_img)
        d_img = np.where(map0, 0, d_img)

        map1 = np.logical_and(t_img == 0, crop_window[:, :, 1:33] >= config.HU_thred)
        d_img = np.where(map1, 1, d_img)

        d_img = d_img.astype('float32')
        d_img /= (config.len_depth - 1)
        d_img = 1.0 - d_img

        if np.sum(d_img) > config.lung_max * config.input_rows * config.input_cols * config.input_deps:
            continue

        img = crop_window[:, :, :config.input_deps]
        img_b = img.astype('float32').tobytes()
        key = hashlib.md5(img_b).hexdigest() + '.' + key_prefix
        lmdb.insert_image(key, img)
        num_pair += 1
        if num_pair == config.scale:
            break


def get_self_learning_data(config, lmdb: lmdbdataset.lmdbdataset):
    luna_subset_path = os.path.join(config.DATA_DIR, )
    file_list = glob(os.path.join(luna_subset_path, "*.nii"))
    for img_file in tqdm(file_list, file=sys.stdout):

        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        img_array = img_array.transpose((2, 1, 0))

        if img_array.shape[0] != 512 or img_array.shape[1] != 512:
            print(f"Invalid: {img_file}")
            continue
        infinite_generator_from_one_volume(config, img_array, os.path.split(img_file)[1], lmdb)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    sys.setrecursionlimit(40000)

    parser = OptionParser()

    parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
    parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
    parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
    parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
    parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
    parser.add_option("--data", dest="data", help="the directory of LUNA16 dataset", default="/dataset/nii_ori",
                      type="string")
    parser.add_option("--save", dest="save", help="the directory of processed 3D cubes", default=None, type="string")
    parser.add_option("--scale", dest="scale", help="scale of the generator", default=32, type="int")
    (options, args) = parser.parse_args()

    seed = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random.seed(seed)
    config = setup_config(input_rows=options.input_rows,
                          input_cols=options.input_cols,
                          input_deps=options.input_deps,
                          crop_rows=options.crop_rows,
                          crop_cols=options.crop_cols,
                          scale=options.scale,
                          len_border=100,
                          len_border_z=30,
                          len_depth=3,
                          lung_min=0.7,
                          lung_max=0.15,
                          DATA_DIR=options.data,
                          )
    config.display()

    lmdbd = lmdbdataset.lmdbdataset('/dataset/lmdb/hospital_64x64x32.lmdb',
                                    (options.input_rows, options.input_cols, options.input_deps))

    get_self_learning_data(config, lmdbd)

    print(lmdbd.key_list())
    print(len(lmdbd.key_list()))
    print(lmdbd.fn_list())
    print(len(lmdbd.fn_list()))
    img = lmdbd.read_image(lmdbd.key_list()[33])
    print(np.shape(img))
    lmdbd.close()
