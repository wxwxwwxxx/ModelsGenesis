import SimpleITK as sitk
import glob
import os

root = r"/dataset/nii_subset/subset2"
f_list = glob.glob(os.path.join(root, "*.nii"))

for root in f_list:

    itk_img = sitk.ReadImage(root)
    sp = itk_img.GetSpacing()
    dr = itk_img.GetDirection()
    img_array = sitk.GetArrayFromImage(itk_img)
    # print("Spacing:", sp)
    # print("Direction:", dr)
    # print("Pre-reshape:",img_array.shape)
    img_array = img_array.transpose(2, 1, 0)
    # print("Post-reshape:",img_array.shape)
    if img_array.shape[0] != 512 or img_array.shape[1] != 512:
        print(f"=========={root}==========")
        print("Spacing:", sp)
        print("Direction:", dr)
        print("Post-reshape:",img_array.shape)