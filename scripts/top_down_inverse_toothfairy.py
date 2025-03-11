import os
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np 

import SimpleITK as sitk
import numpy as np

import scipy.ndimage

# add multiprocessing
from multiprocessing import Pool
from functools import partial


def rotate_file(filename, root, save_path):
    file_path = os.path.join(root, filename)
    image = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(image)  # Convert to NumPy array (shape: [D, H, W])

    # Rotate 180 degrees along the Anterior-Posterior (A-P) axis (Y-axis)
    rotated_data = scipy.ndimage.rotate(data, 180, axes=(0, 2), reshape=False, mode='nearest')

    # make sure data type still the same
    rotated_data = rotated_data.astype(data.dtype)
    rotated_image = sitk.GetImageFromArray(rotated_data)

    rotated_image.CopyInformation(image)  # Preserve metadata

    # Save the rotated image
    save_file_path = os.path.join(save_path, filename)
    sitk.WriteImage(rotated_image, save_file_path)



dataset_dirs = ['/usr/bmicnas02/data-biwi-01/ct_video_mae/ct_datasets/toothfairy/new_version/Dataset112_ToothFairy2/imagesTr_org/','/usr/bmicnas02/data-biwi-01/ct_video_mae/ct_datasets/toothfairy/new_version/Dataset112_ToothFairy2/labelsTr_org/']




for dataset_dir in dataset_dirs:
    all_files = os.listdir(dataset_dir)
    all_files = [f for f in all_files if f.endswith('.mha')]
    all_files = sorted(all_files)

    save_path = dataset_dir.replace('org','rotated')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # using multiprocessing
    with Pool(2) as p:
        p.map(partial(rotate_file, root=dataset_dir, save_path=save_path), all_files)





# # Load the .mha file
# file_path = "/usr/bmicnas02/data-biwi-01/ct_video_mae/ct_datasets/toothfairy/new_version/Dataset112_ToothFairy2/imagesTr/ToothFairy2F_001_0000.mha"
# image = sitk.ReadImage(file_path)
# data = sitk.GetArrayFromImage(image)  # Convert to NumPy array (shape: [D, H, W])

# # Rotate 180 degrees along the Anterior-Posterior (A-P) axis (Y-axis)


# # Convert back to SimpleITK image
# rotated_data = scipy.ndimage.rotate(data, 180, axes=(0, 2), reshape=False, mode='nearest')
# rotated_image = sitk.GetImageFromArray(rotated_data)

# rotated_image.CopyInformation(image)  # Preserve metadata

# # Save the rotated image
# sitk.WriteImage(rotated_image, "./rotated_180_AP.mha")

