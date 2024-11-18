import os
import shutil
import json 
import copy
import numpy as np 

images_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/image'
tgt_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/rename_data'

images_name_list = sorted(os.listdir(images_path))
images_name_list = [x for x in images_name_list if x.endswith('.nii.gz')]

for i, image_name in enumerate(images_name_list):
    image_name_no_ext = image_name.split('.nii.gz')[0]
    os.makedirs(os.path.join(tgt_path, image_name_no_ext), exist_ok=True)
    new_name = image_name_no_ext + f'_0000.nii.gz'
    image_path = os.path.join(images_path, image_name)
    new_image_path = os.path.join(tgt_path, new_name)
    shutil.copy(image_path, new_image_path)
    print("copy file from", image_path , 'to', new_image_path)
