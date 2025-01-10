import os
import shutil
import json 
import copy
import numpy as np 

images_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/imagesTr'
masks_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/labelsTr'
split_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/splits_final.json'

new_images_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/imagesTr'
new_masks_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/labelsTr'
new_split_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/splits_final.json'
old_to_new_mappping_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/old_to_new_mapping.json'

with open(split_path, 'r') as f:
    split_file = json.load(f)

print(split_file[0].keys())
# train_split = split_file[0]['train']
val_split = split_file[0]['val']
copy_split = copy.deepcopy(split_file[0])
copy_split['train'] = []
copy_split['val'] = []

num_count = 0   
images_name_list = sorted(os.listdir(images_path))
print("images_name_list", len(images_name_list))

old_to_new_mappping = {}
old_to_new_mappping['train'] = {}
old_to_new_mappping['val'] = {}

for i, image_name in enumerate(images_name_list):
    image_path = os.path.join(images_path, image_name)
    vol_name = image_name.split('_0000.nii.gz')[0]
    mask_name = vol_name + '.nii.gz'
    mask_path = os.path.join(masks_path, mask_name)

    assert os.path.exists(image_path), f'{image_path} does not exist'
    assert os.path.exists(mask_path), f'{mask_path} does not exist'


    # renameing the file
    new_image_name = f'{i:04d}_0000.nii.gz'
    new_mask_name = f'{i:04d}.nii.gz'
    new_image_path = os.path.join(new_images_path, new_image_name)
    new_mask_path = os.path.join(new_masks_path, new_mask_name)

    # shutil.copy(image_path, new_image_path)
    # shutil.copy(mask_path, new_mask_path)
    print("copy file from", image_path , 'to', new_image_path)
    print("copy mask from", mask_path , 'to', new_mask_path)

    if vol_name in val_split:
        # copy_split['val'].remove(vol_name)
        copy_split['val'].append(f'{i:04d}')
        old_to_new_mappping['val'][vol_name] = f'{i:04d}'
    else:
        copy_split['train'].append(f'{i:04d}')

        old_to_new_mappping['train'][vol_name] = f'{i:04d}'

    # update the split
    # if vol_name in train_split:
    #     copy_split['train'].remove(vol_name)
    #     copy_split['train'].append(f'{num_count:04d}')
    # elif vol_name in val_split:
    #     copy_split['val'].remove(vol_name)
    #     copy_split['val'].append(f'{num_count:04d}')
    # else:
    #     raise ValueError(f'{vol_name} not in train or val')

# remove redundant one in train and val
# copy_split['train'] = np.unique(copy_split['train'])
# copy_split['val'] =  np.unique(copy_split['val'])

print("copy_split", copy_split['train'], len(copy_split['train']))
print("copy_split", copy_split['val'], len(copy_split['val']))


with open(old_to_new_mappping_path, 'w') as f:
    json.dump(old_to_new_mappping, f, indent=4)
# save the new split
# with open(new_split_path, 'w') as f:
#     json.dump(copy_split, f, indent=4)