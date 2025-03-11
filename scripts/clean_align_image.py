import os
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np 
import json
from multiprocessing import Pool
from functools import partial
import SimpleITK as stik


def modify_main_func(data_path):
    # load data
    data = stik.ReadImage(data_path)
    img = stik.GetArrayFromImage(data)
    # # remove label > 55
    # img[img > 55] = 0
    # clip to -1000 to 4000
    img = np.clip(img, -1000, 4000)
    # save data with previous info
    modify_data = stik.GetImageFromArray(img)
    modify_data.CopyInformation(data)
    stik.WriteImage(modify_data, data_path)


labels_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/imagesTs'

# '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset121_ToothFairy2_to_Align_new_labels/imagesTr'
# '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/imagesTr'

data = os.listdir(labels_dir)
data = [d for d in data if d.endswith('.nii.gz')]
# data_idx_dict = {}

print("cleaning data", labels_dir, "len", len(data))
with Pool(4) as p:
    p.map(partial(modify_main_func), [os.path.join(labels_dir, data_i) for data_i in data])

# for data_i in tqdm(data):
#     # load data
#     data_i_path = os.path.join(labels_dir, data_i)
#     data_i_nii = stik.ReadImage(data_i_path)
#     data_i_data = stik.GetArrayFromImage(data_i_nii)
#     # remove label > 55
#     data_i_data[data_i_data > 55] = 0
#     # save data with previous info
#     modify_data_i_nii = stik.GetImageFromArray(data_i_data)
#     modify_data_i_nii.CopyInformation(data_i_nii)
#     stik.WriteImage(modify_data_i_nii, data_i_path)

    # data_i_nii = nib.load(data_i_path)
    # data_i_data = data_i_nii.get_fdata()
    # # remove label > 55
    # data_i_data[data_i_data > 55] = 0
    # # save data with previous info

    # unique_labels = np.unique(data_i_data)
    # print("unique_labels",  unique_labels)
    # for unique_label_j in unique_labels:
    #     if unique_label_j in data_idx_dict.keys():
    #         data_idx_dict[unique_label_j].append(data_i)
    #     else:
    #         data_idx_dict[unique_label_j] = [data_i]

    # with open("./label_mapping_new.json", "w") as f:
    #     json.dump(data_idx_dict, f, indent=4)  # `indent=4` makes it more readable


# with open('label_count.txt', 'w') as f:
#     for key, value in data_idx_dict.items():
#         f.write(f"{key}: {value}\n")




# labels_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/labelsTr/'
# save_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/labelsTr_filter/'

# data = os.listdir(labels_dir)
# data = [d for d in data if d.endswith('.nii.gz')]
# data_idx_dict = {}

# for data_i in tqdm(data):
#     # load data
#     data_i_path = os.path.join(labels_dir, data_i)
#     data_i_nii = nib.load(data_i_path)
#     data_i_data = data_i_nii.get_fdata()
#     # mv supplmentary teeth to 0



