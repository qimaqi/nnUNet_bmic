from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from tqdm import tqdm
from os.path import join
import json
import os 
import SimpleITK as sitk

def convert_align(align_base_dir: str, nnunet_dataset_id: int = 220):
    task_name = "teeth"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests= join(out_base, "imagesTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)

    old_to_new_mapping_json = join(out_base, "old_to_new_mapping.json")
    print("old_to_new_mapping_json", old_to_new_mapping_json)
    with open(old_to_new_mapping_json, 'r') as f:
        old_to_new_mapping = json.load(f)
        
    image_path = join(align_base_dir, 'imagesTr')
    label_path = join(align_base_dir, 'labelsTr')
    for image_i in os.listdir(image_path):
        if image_i.endswith('.nrrd'):
            image_i_base = image_i.split('.')[0]
            mask_i_base = image_i.replace('_0000.nrrd', '.nrrd')
            image_i_path = join(image_path, image_i)
            mask_i_path = join(label_path, mask_i_base)
            assert isfile(image_i_path), image_i_path
            assert isfile(mask_i_path), mask_i_path

            old_name_base = image_i_base.replace('_0000', '')
            if old_name_base in old_to_new_mapping['train']:
                new_image_base = old_to_new_mapping['train'][old_name_base]
                new_image_i = new_image_base + '_0000.nii.gz'
                new_mask_i = new_image_base + '.nii.gz'
                new_image_i_save_path = join(imagestr, new_image_i)
                new_mask_i_save_path = join(labelstr, new_mask_i)

            elif old_name_base in old_to_new_mapping['val']:
                new_image_base = old_to_new_mapping['val'][old_name_base]
                new_image_i = new_image_base + '_0000.nii.gz'
                new_mask_i = new_image_base + '.nii.gz'
                new_image_i_save_path = join(imagests, new_image_i)
                new_mask_i_save_path = join(labelstr, new_mask_i)

            print("convert", image_i_path, "to", new_image_i_save_path)
            data = sitk.ReadImage(image_i_path)
            # save
            sitk.WriteImage(data, new_image_i_save_path)

            print("convert", mask_i_path, "to", new_mask_i_save_path)
            data = sitk.ReadImage(mask_i_path)
            # save
            sitk.WriteImage(data, new_mask_i_save_path)


            # new_image_i = old_to_new_mapping[image_i_base] + '.nii.gz'
            # shutil.copy(join(image_path, image_i), join(imagestr, new_image_i))
            # shutil.copy(join(label_path, image_i), join(labelstr, new_image_i))


    # for tr in tqdm(cases):
    #     shutil.copy(join(align_base_dir, tr, 'imaging.nii.gz'), join(imagestr, f'{tr}_0000.nii.gz'))
    #     shutil.copy(join(align_base_dir, tr, 'segmentation.nii.gz'), join(labelstr, f'{tr}.nii.gz'))

    # generate_dataset_json(out_base, {0: "CT"},
    #                       labels={
    #                           "background": 0,
    #                           "kidney": (1, 2, 3),
    #                           "masses": (2, 3),
    #                           "tumor": 2
    #                       },
    #                       regions_class_order=(1, 3, 2),
    #                       num_training_cases=len(cases), file_ending='.nii.gz',
    #                       dataset_name=task_name, reference='none',
    #                       release='0.1.3',
    #                       overwrite_image_reader_writer='NibabelIOWithReorient',
    #                       description="KiTS2023")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted LiTS dataset (must have .nrrd files)")
    parser.add_argument('-d', required=False, type=int, default=888, help='nnU-Net Dataset ID, default: 888')
    args = parser.parse_args()
    align_base = args.input_folder
    convert_align(align_base, args.d)

    # /media/isensee/raw_data/raw_datasets/kits23/dataset

