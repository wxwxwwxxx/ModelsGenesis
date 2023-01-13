import time
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
import glob
from skimage.transform import resize
import lmdbdataset
import hashlib
import pandas as pd
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
def getcrop_index(arr_shape,crop_shape,index_zyx):
    # arr_shape index:zyx
    np_arr = np.array(arr_shape)
    np_crop = np.array(crop_shape)
    np_index = np.array(index_zyx)
    np_index_min = np_index-(np_crop//2)
    np_index_min = np.where(np_index_min<0,0,np_index_min)
    np_index_min = np.where((np_index_min+np_crop)>np_arr,np_arr-np_crop,np_index_min)
    np_index_max = np_index_min+np_crop
    return np_index_min.astype('int32'),np_index_max.astype('int32')
def get_candidate_dict(candidate_csv_path):
    candidate = pd.read_csv(candidate_csv_path)
    candidate_dict = {}
    for i, r in candidate.iterrows():

        if r["seriesuid"] not in candidate_dict:
            candidate_dict[r["seriesuid"]] = []
        candidate_dict[r["seriesuid"]].append((r["coordX"], r["coordY"], r["coordZ"], r["class"]))
    return candidate_dict



if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
    parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
    parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
    parser.add_option("--data", dest="data", help="the directory of LUNA16 dataset", default="/dataset/LUNA2016",type="string")
    parser.add_option("--save", dest="save", help="the directory of processed 3D cubes", default=None, type="string")

    (options, args) = parser.parse_args()

    config = setup_config(input_rows=options.input_rows,
                          input_cols=options.input_cols,
                          input_deps=options.input_deps,
                          len_border=100,
                          len_border_z=30,
                          len_depth=3,
                          lung_min=0.7,
                          lung_max=0.15,
                          DATA_DIR=options.data,
                          )
    config.display()
    cut_size=(config.input_deps,config.input_rows,config.input_cols)#ZYX
    candidate_dict = get_candidate_dict(os.path.join(config.DATA_DIR,"candidates.csv"))
    # print(candidate_dict)
    lmdbd = lmdbdataset.lmdbdataset('/dataset/lmdb/luna16_fpr_64x64x32.lmdb',
                                    (options.input_cols, options.input_rows, options.input_deps))
    # lmdbd.
    for sub in range(10):
        fl = glob.glob(os.path.join(config.DATA_DIR,f"subset{sub}","*.mhd"))
        print(f"subset{sub}")
        for i in tqdm(fl):
            key = os.path.split(os.path.splitext(i)[0])[1]
            sitkimg = sitk.ReadImage(i)

            img_array = sitk.GetArrayFromImage(sitkimg)
            # normalization

            img_array[img_array < config.hu_min] = config.hu_min
            img_array[img_array > config.hu_max] = config.hu_max
            img_array = 1.0 * (img_array - config.hu_min) / (config.hu_max - config.hu_min)
            arr_shape = img_array.shape

            for g_coord in candidate_dict[key]:

                    coord = sitkimg.TransformPhysicalPointToIndex((g_coord[0], g_coord[1], g_coord[2]))

                    ret0, ret1 = getcrop_index(arr_shape, cut_size, (coord[2], coord[1], coord[0],))

                    sub_array = img_array[ret0[0]:ret1[0], ret0[1]:ret1[1], ret0[2]:ret1[2]].transpose((2,1,0))
                    assert sub_array.shape == (64,64,32)
                    img_b = sub_array.astype('float32').tobytes()
                    key = hashlib.md5(img_b).hexdigest() + '_' + f"subset{sub}" + '_' + f"{g_coord[3]}"
                    # print(key)
                    lmdbd.insert_image_byte(key,img_b)