from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm

if __name__ == '__main__':
    """
    How to train our submission to the JHU benchmark
    
    1. Execute this script here to convert the dataset into nnU-Net format. Adapt the paths to your system!
    2. Run planning and preprocessing: `nnUNetv2_plan_and_preprocess -d 224 -npfp 64 -np 64 -c 3d_fullres -pl 
    nnUNetPlannerResEncL_torchres`. Adapt the number of processes to your System (-np; -npfp)! Note that each process 
    will again spawn 4 threads for resampling. This custom planner replaces the nnU-Net default resampling scheme with 
    a torch-based implementation which is faster but less accurate. This is needed to satisfy the inference speed 
    constraints.
    3. Run training with `nnUNetv2_train 224 3d_fullres all -p nnUNetResEncUNetLPlans_torchres`. 24GB VRAM required, 
    training will take ~28-30h.
    """


    base = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/abdomatlas01'
    cases = subdirs(base, join=False, prefix='BDMAP')
    exist_cases = []
    # check if ct.nii.gz and combined_labels.nii.gz exist in each case folder
    for case in cases:
        if os.path.exists(join(base, case, 'ct.nii.gz')) and os.path.exists(join(base, case, 'combined_labels.nii.gz')):
            print(f"Case {case} is complete")
            exist_cases.append(case)
        else:
            print(f"Case {case} is incomplete")
            # remove case from list

    cases = exist_cases

    target_dataset_id = 224
    target_dataset_name = f'Dataset{target_dataset_id:3.0f}_AbdomenAtlas1.0'

    raw_dir = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw'
    maybe_mkdir_p(join(raw_dir, target_dataset_name))
    imagesTr = join(raw_dir, target_dataset_name, 'imagesTr')
    labelsTr = join(raw_dir, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    for case in tqdm(cases):
        if os.path.exists(join(imagesTr, case + '_0000.nii.gz')):
            print(f"Case {case} already processed, skipping")
            continue

        shutil.copy(join(base, case, 'ct.nii.gz'), join(imagesTr, case + '_0000.nii.gz'))
        shutil.copy(join(base, case, 'combined_labels.nii.gz'), join(labelsTr, case + '.nii.gz'))

    labels = {
        "background": 0,
        "aorta": 1,
        "gall_bladder": 2,
        "kidney_left": 3,
        "kidney_right": 4,
        "liver": 5,
        "pancreas": 6,
        "postcava": 7,
        "spleen": 8,
        "stomach": 9
    }

    generate_dataset_json(
        join(raw_dir, target_dataset_name),
        {0: 'nonCT'},  # this was a mistake we did at the beginning and we keep it like that here for consistency
        labels,
        len(cases),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient'
    )