import os
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np 
labels_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset224_AbdomenAtlas1.0/gt_segmentations/'
old_path_list = os.listdir(labels_dir)

# count labels over 10 and number
num_over_10_list = []
for label_i in tqdm(old_path_list):
    label_i_path = os.path.join(labels_dir, label_i)
    label_i_nii = nib.load(label_i_path)
    label_i_data = label_i_nii.get_fdata()

    # print unique label
    unique_labels = np.unique(label_i_data)
    print(f"Unique labels in {label_i}: {unique_labels}")

    # num_over_10 = np.sum(label_i_data >= 10)
    # num_over_10_list.append(num_over_10)
    # if num_over_10 > 0:
    #     print(f"Label {label_i} has {num_over_10} labels over 10.")
    #     mask = label_i_data >= 10
    #     labels_over_10 = label_i_data[mask]
    #     # find unique labels over 10
    #     unique_labels = np.unique(labels_over_10)
    #     print(f"Unique labels over 10: {unique_labels}")


print(f"Number of labels over 10: {np.sum(num_over_10_list)}")


