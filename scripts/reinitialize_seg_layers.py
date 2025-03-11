import os
import shutil
import nibabel as nib
from tqdm import tqdm
import numpy as np 

ckpt_path = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_results/Dataset888_teeth/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth'

reinitialize_path = [
    '.seg_layers.',
]
# skip seg_layers means last layer is always reinitialized





# model_dict = mod.state_dict()
# # verify that all but the segmentation layers have the same shape
# if use_nnunet:
# for key, _ in model_dict.items():
#     if all([i not in key for i in skip_strings_in_pretrained]):
#         import json
#         assert key in pretrained_dict, \
#             f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
#             f"compatible with your network."
        
#         assert model_dict[key].shape == pretrained_dict[key].shape, \
#             f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
#             f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
#             f"does not seem to be compatible with your network."