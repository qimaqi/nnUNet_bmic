#!/bin/bash
#SBATCH --job-name=resize_slices
#SBATCH --account=staff 
#SBATCH --constraint='titan_xp'
#SBATCH --output=sbatch_log/mae_resize_pretrain_full_%j.out
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:5

# Load any necessary modules
source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate nnunet_env

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export PATH=/scratch_net/schusch/qimaqi/install_gcc:$PATH
export CC=/scratch_net/schusch/qimaqi/install_gcc/bin/gcc-11.3.0
export CXX=/scratch_net/schusch/qimaqi/install_gcc/bin/g++-11.3.0

export nnUNet_raw="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw"
export nnUNet_preprocessed="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed"
export nnUNet_results="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_results"




nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_extract_fingerprint -d 119 -np 16 
nnUNetv2_plan_experiment -d 119 -pl nnUNetPlannerResEncL_torchres
nnUNetv2_preprocess -d 119 -plans_name nnUNetResEncUNetLPlans_torchres -np 16

nnUNetv2_train 119 3d_fullres all -p nnUNetResEncUNetLPlans_torchres