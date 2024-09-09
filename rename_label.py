import os
import shutil

labels_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset000_Toothfairy/labelsTr'
old_path_list = os.listdir(labels_dir)


for label_i in old_path_list:
    label_i_path = os.path.join(labels_dir, label_i)
    new_label_i = label_i.replace('_0000.nii.gz', '.nii.gz')
    new_label_i_path = os.path.join(labels_dir, new_label_i)
    shutil.move(label_i_path, new_label_i_path)
