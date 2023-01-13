import os
import shutil
import random
import SimpleITK as sitk
## distribute

# file_src = os.path.join('/dataset/nii_ori')
# file_target = os.path.join('/dataset/nii_subset')
# patient_dict = {}
# fold = 10
# subset_dir = [os.path.join(file_target,"subset"+str(i)) for i in range(fold)]
# for i in subset_dir:
#     os.makedirs(i,exist_ok=True)
# invalid_dir = os.path.join(file_target,"invalid")
# os.makedirs(invalid_dir,exist_ok=True)
#
# for fn in os.listdir(file_src):
#     full_path = os.path.join(file_src,fn)
#     #print(full_path)
#
#     itk_img = sitk.ReadImage(full_path)
#     img_array = sitk.GetArrayFromImage(itk_img)
#     img_array = img_array.transpose(2, 1, 0)
#     if img_array.shape[0] != 512 or img_array.shape[1] != 512:
#         print(f"Invalid: {full_path}")
#         shutil.move(full_path, invalid_dir)
#         continue
#
#     fn = fn.split('.')
#     patient = fn[0]
#     if patient in patient_dict:
#         shutil.move(full_path, subset_dir[patient_dict[patient]])
#         print(full_path,invalid_dir)
#         continue
#
#     rand_target = random.randint(0, 9)
#     patient_dict[patient] = rand_target
#     print(full_path, subset_dir[rand_target])
#     shutil.move(full_path, subset_dir[rand_target])

## gather

file_src = os.path.join('/dataset/nii_ori')
file_target = os.path.join('/dataset/nii_subset')

for f_root,f_dir,f_f in os.walk(file_target):

    for f in f_f:

        full_path = os.path.join(f_root, f)
        # print(full_path)
        # fn = fn.split('.')
        # ac = fn[0]
        # if ac in ac_set:
        #     shutil.move(full_path, r_dir)
        #     # print(full_path,r_dir)
        #     continue
        # ac_set.add(ac)
        # rand_target = random.randint(0,9)
        print(full_path, file_src)
        if not os.path.exists(os.path.join(file_src,f)):
            shutil.move(full_path, file_src)






