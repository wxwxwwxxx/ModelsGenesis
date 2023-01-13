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
def numpy_norm(img):
    img = img.astype("float32")
    img_min = np.min(img)
    img_max = np.max(img)
    img_array = 1.0 * (img - img_min) / (img_max - img_min)
    return img_array
class setup_config():
    hu_max = 1000.0
    hu_min = -1000.0
    HU_thred = (-800 - hu_min) / (hu_max - hu_min)

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
    np_index_min = np_index
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

    parser.add_option("--input_rows", dest="input_rows", help="input rows", default=240, type="int")
    parser.add_option("--input_cols", dest="input_cols", help="input cols", default=240, type="int")
    parser.add_option("--input_deps", dest="input_deps", help="input deps", default=155, type="int")
    parser.add_option("--data", dest="data", help="the directory of LUNA16 dataset", default="/dataset/BraTS/MICCAI_BraTS_2018_Data_Training",type="string")
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
    lmdbd = lmdbdataset.lmdbdataset('/dataset/lmdb/brats_240x240x155.lmdb',
                                    (options.input_cols, options.input_rows, options.input_deps))
    for fn,fd,ff in os.walk(config.DATA_DIR):
        if len(ff)!=5:
            continue
        test_name = fn.split('/')[-1]
        #order=(t1,t2,t1ce,flair,seg)
        t1 = sitk.ReadImage(os.path.join(fn,f"{test_name}_t1.nii.gz"))
        t1ce = sitk.ReadImage(os.path.join(fn,f"{test_name}_t1ce.nii.gz"))
        t2 = sitk.ReadImage(os.path.join(fn,f"{test_name}_t2.nii.gz"))
        flair = sitk.ReadImage(os.path.join(fn,f"{test_name}_flair.nii.gz"))

        seg = sitk.ReadImage(os.path.join(fn,f"{test_name}_seg.nii.gz"))
        print(f"=={test_name}")
        t1_arr = numpy_norm(sitk.GetArrayFromImage(t1).transpose((2,1,0)))
        t1ce_arr = numpy_norm(sitk.GetArrayFromImage(t1ce).transpose((2,1,0)))
        t2_arr = numpy_norm(sitk.GetArrayFromImage(t2).transpose((2,1,0)))
        flair_arr = numpy_norm(sitk.GetArrayFromImage(flair).transpose((2,1,0)))

        seg_arr = sitk.GetArrayFromImage(seg).transpose((2,1,0)).astype('int32')
        seg_arr = np.where(seg_arr != 0, 1, 0)

        key_t1 = f"{test_name}_t1"
        key_t1ce = f"{test_name}_t1ce"
        key_t2 = f"{test_name}_t2"
        key_flair = f"{test_name}_flair"

        key_seg = f"{test_name}_seg"

        lmdbd.insert_image(key_t1, t1_arr)
        lmdbd.insert_image(key_t1ce, t1ce_arr)
        lmdbd.insert_image(key_t2, t2_arr)
        lmdbd.insert_image(key_flair, flair_arr)
        lmdbd.insert_image(key_seg, seg_arr)
        # img_arr = np.stack([t1_arr,t1ce_arr,t2_arr,flair_arr],axis=-1)
        # print(img_arr.shape)

    # # lmdbd.
    # for i in tqdm(range(131)):
    #     img_fn = os.path.join(config.DATA_DIR, f"volume-{i}.nii")
    #     seg_fn = os.path.join(config.DATA_DIR, f"segmentation-{i}.nii")
    #
    #     sitkimg = sitk.ReadImage(img_fn)
    #     sitkseg = sitk.ReadImage(seg_fn)
    #
    #     # seg_fn = glob.glob(os.path.join(config.DATA_DIR,"volume-*"))
    #     # print(index)
    #     img_array = sitk.GetArrayFromImage(sitkimg)
    #
    #     seg_array = sitk.GetArrayFromImage(sitkseg).astype('uint8')
    #     seg_array = np.where(seg_array != 1, 0, 1)
    #
    #     img_array[img_array < config.hu_min] = config.hu_min
    #     img_array[img_array > config.hu_max] = config.hu_max
    #     img_array = 1.0 * (img_array - config.hu_min) / (config.hu_max - config.hu_min)
    #
    #     arr_shape = img_array.shape
    #     count = 0
    #     while count < 50:
    #
    #         z = random.randint(0,arr_shape[0])
    #         y = random.randint(0,arr_shape[1])
    #         x = random.randint(0,arr_shape[2])
    #         ret0, ret1 = getcrop_index(arr_shape, cut_size, (z,y,x))
    #         img_sub_arr = img_array[ret0[0]:ret1[0], ret0[1]:ret1[1], ret0[2]:ret1[2]].transpose((2, 1, 0))
    #         seg_sub_arr = seg_array[ret0[0]:ret1[0], ret0[1]:ret1[1], ret0[2]:ret1[2]].transpose((2, 1, 0))
    #         assert img_sub_arr.shape == (options.input_cols, options.input_rows, options.input_deps)
    #         assert seg_sub_arr.shape == (options.input_cols, options.input_rows, options.input_deps)
    #         if np.sum(seg_sub_arr) == 0:
    #             continue
    #         key_img = f"patch{count}" + '_' + f"{i}" + '_img'
    #         key_seg = f"patch{count}" + '_' + f"{i}" + '_seg'
    #         lmdbd.insert_image(key_img, img_sub_arr)
    #         lmdbd.insert_image(key_seg, seg_sub_arr)
    #         count += 1

        # for g_coord in candidate_dict[key]:
        #
        #         coord = sitkimg.TransformPhysicalPointToIndex((g_coord[0], g_coord[1], g_coord[2]))
        #
        #         ret0, ret1 = getcrop_index(arr_shape, cut_size, (coord[2], coord[1], coord[0],))
        #
        #         sub_array = img_array[ret0[0]:ret1[0], ret0[1]:ret1[1], ret0[2]:ret1[2]].transpose((2,1,0))
        #         assert sub_array.shape == (64,64,32)
        #         img_b = sub_array.astype('float32').tobytes()
        #         key = hashlib.md5(img_b).hexdigest() + '_' + f"subset{sub}" + '_' + f"{g_coord[3]}"
        #         # print(key)
        #         lmdbd.insert_image_byte(key,img_b)