#!/usr/bin/env python
# coding: utf-8

import warnings
import datetime
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
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
import time
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from infinite_generator_3D import setup_config, infinite_generator_from_one_volume


def get_self_learning_data_multi_process(file_list, config, lmdb_path, shape):
    lmdb = lmdbdataset.lmdbdataset(lmdb_path,shape,lock=True)

    for img_file in tqdm(file_list, file=sys.stdout):
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        img_array = img_array.transpose((2, 1, 0))

        if img_array.shape[0] != 512 or img_array.shape[1] != 512:
            # print(f"Invalid: {img_file}\n",end="")
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
    conf = setup_config(input_rows=options.input_rows,
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
                          process_num=cpu_count()
                          )

    print(f"CPU Count: {cpu_count()}")

    luna_subset_path = os.path.join(conf.DATA_DIR, )
    file_list = glob(os.path.join(luna_subset_path, "*.nii"))
    print(f"NII Count: {len(file_list)}")
    sub_len = math.ceil(len(file_list) / conf.process_num)
    sub_file_list = [file_list[i*sub_len:(i+1)*sub_len] for i in range(conf.process_num)]


    args_list = [(i,conf,"/dataset/lmdb/hospital_646432.lmdb",(options.input_rows,options.input_cols,options.input_deps)) for i in sub_file_list]

    pool = Pool(processes=conf.process_num)
    pool.starmap(get_self_learning_data_multi_process, args_list)
    print("stop")

