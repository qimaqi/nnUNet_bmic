import os
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np 
import SimpleITK as sitk
# example_data0 = '/usr/bmicnas02/data-biwi-01/ct_video_mae/ct_datasets/toothfairy/new_version/Dataset112_ToothFairy2/imagesTr/ToothFairy2F_002_0000.mha'

# example_data1 = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/imagesTr/0000_0000.nii.gz'


# # data0 = nib.load(example_data0)
# data0 = sitk.ReadImage(example_data0)
# data1 = nib.load(example_data1)

# # data0 = data0.get_fdata()
# data0 = sitk.GetArrayFromImage(data0)
# data1 = data1.get_fdata()

# print(data0.shape, data1.shape)

# print("Data0 range: ", np.min(data0), np.max(data0))
# print("Data1 range: ", np.min(data1), np.max(data1))


processed_align = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset121_ToothFairy2_to_Align_new_labels/nnUNetPlans_3d_fullres/ToothFairy2F_055.npz'
# '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset120_ToothFairy2_to_Align/nnUNetPlans_3d_fullres/ToothFairy2F_005.npz'
# '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/nnUNetPlans_3d_fullres/0016.npz'
# '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/nnUNetPlans_3d_fullres/0000.npz'
# '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/nnUNetPlans_3d_fullres/0000.npy'

# convert to nii.gz
processed_align_data = np.load(processed_align)['data']
print(processed_align_data.shape)
print("Processed align range: ", np.min(processed_align_data), np.max(processed_align_data))
import nibabel as nib
nib_data = nib.Nifti1Image(processed_align_data[0], np.eye(4))
nib.save(nib_data, './fairy.nii.gz')

processed_align_seg = np.load(processed_align)['seg']
processed_align_seg = np.array(processed_align_seg)
print(processed_align_seg.shape)
nib_data = nib.Nifti1Image(processed_align_seg[0], np.eye(4))
nib.save(nib_data, './fairy_seg.nii.gz')