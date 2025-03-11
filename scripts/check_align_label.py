import os
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np 
import json
from multiprocessing import Pool
from functools import partial

labels_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/labelsTr'

data = os.listdir(labels_dir)
data = [d for d in data if d.endswith('.nii.gz')]
data_idx_dict = {}

for data_i in tqdm(data):
    # load data
    data_i_path = os.path.join(labels_dir, data_i)
    data_i_nii = nib.load(data_i_path)
    data_i_data = data_i_nii.get_fdata()
    unique_labels = np.unique(data_i_data)
    print("unique_labels",  unique_labels)
    for unique_label_j in unique_labels:
        if unique_label_j in data_idx_dict.keys():
            data_idx_dict[unique_label_j].append(data_i)
        else:
            data_idx_dict[unique_label_j] = [data_i]

    with open("./label_mapping_new.json", "w") as f:
        json.dump(data_idx_dict, f, indent=4)  # `indent=4` makes it more readable


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



