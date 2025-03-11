import os
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np 
import SimpleITK as sitk

def normalize_hist(image: sitk.Image, th=0.999) -> sitk.Image:
    arr = sitk.GetArrayViewFromImage(image)
    # print min and high
    print("intensity min,", arr.min(), 'max', arr.max())
    print("intesntiy p1", np.percentile(arr, 1), 'p99', np.percentile(arr, 99))
    # manuual cut to -1000-2500
    # arr = arr.clip(-1000, 2500)
    ns, intensity = np.histogram(arr.reshape(-1), bins=256)

    cutoff_1 = np.ediff1d(ns[:-1]).argmax(axis=0) + 1
    total = np.sum(ns[cutoff_1 + 1:-1])
    cutoff_2 = (np.cumsum(ns[cutoff_1 + 1:].astype(int) / total) > th).argmax() + cutoff_1

    # print("cutoff1 intensity: ", intensity[cutoff_1], cutoff_1)
    # print("cutoff2 intensity: ", intensity[cutoff_2], cutoff_2)

    # image = sitk.Clamp(image, outputPixelType=sitk.sitkFloat32,
    #                    lowerBound=float(intensity[cutoff_1]), upperBound=float(intensity[cutoff_2]))
    image = sitk.Clamp(image, outputPixelType=sitk.sitkFloat32,
                       lowerBound=float(intensity[0]), upperBound=float(intensity[cutoff_2]))

    return sitk.RescaleIntensity(image)




dataset_dir = '/usr/bmicnas02/data-biwi-01/ct_video_mae/ct_datasets/toothfairy/new_version/Dataset112_ToothFairy2/imagesTr_rotated/'
save_path = '/usr/bmicnas02/data-biwi-01/ct_video_mae/ct_datasets/toothfairy/new_version/Dataset112_ToothFairy2/imagesTr_norm/'


all_files = os.listdir(dataset_dir)
all_files = [f for f in all_files if f.endswith('.mha')]
all_files = sorted(all_files)

# save_path = dataset_dir.replace('org','rotated')
if not os.path.exists(save_path):
    os.makedirs(save_path)

for file_i in tqdm(all_files):
    file_path = os.path.join(dataset_dir, file_i)
    image = sitk.ReadImage(file_path)
    image = normalize_hist(image)
    save_file_path = os.path.join(save_path, file_i)
    sitk.WriteImage(image, save_file_path)
