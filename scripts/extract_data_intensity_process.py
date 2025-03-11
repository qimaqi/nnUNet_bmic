import os
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np 
import SimpleITK as sitk
# multiprocessing
import multiprocessing
from joblib import Parallel, delayed

processed_data_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/nnUNetPlans_3d_fullres'
raw_label_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/gt_segmentations'
json_save_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/data_info'

if not os.path.exists(json_save_path):
    os.makedirs(json_save_path)

data_list = os.listdir(processed_data_path)
data_list = [x for x in data_list if x.endswith('.npz')]
data_list = sorted(data_list)

for data_name_i in tqdm(data_list):
    data_path = os.path.join(processed_data_path, data_name_i)
    # data = nib.load(data_path)
    data_npz = np.load(data_path)
    data = data_npz['data']
    # label_path = os.path.join(raw_label_path, data_name_i.replace('npz', 'nii.gz'))
    # label = nib.load(label_path)
    label = data_npz['seg']
    # data = data.get_fdata()
    # label = label.get_fdata()

    data_info = {}
    data_info['shape'] = data.shape
    data_info['min'] = np.min(data)
    data_info['max'] = np.max(data)
    for label_id in np.unique(label):
        if len(label[label==label_id]) == 0:
            continue
        data_info['label_'+str(label_id)] = {}
        data_info['label_'+str(label_id)]['min'] = np.min(data[label==label_id])
        data_info['label_'+str(label_id)]['max'] = np.max(data[label==label_id])
        data_info['label_'+str(label_id)]['mean'] = np.mean(data[label==label_id])
        data_info['label_'+str(label_id)]['std'] = np.std(data[label==label_id])
        data_info['label_'+str(label_id)]['p99'] = np.percentile(data[label==label_id], 99)
        data_info['label_'+str(label_id)]['p01'] = np.percentile(data[label==label_id], 1) 

    # save info to .npy file
    print("data_info: ", data_info)
    np.save(os.path.join(json_save_path, data_path.split('/')[-1].replace('.nii.gz', '.npy')), data_info)


    # np.save(os.path.join(json_save_path, data_path.split('/')[-1].replace('.nii.gz', '.npy')), data_info)
    # # print(data.shape)