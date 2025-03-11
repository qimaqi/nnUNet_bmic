#!/bin/bash
#SBATCH --job-name=n120_3d_fullres
#SBATCH --output=sbatch_log/n120_3d_videomae_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu07,bmicgpu08,bmicgpu09,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.eins.qi@gmail.com

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



# toothfairy
# nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
# nnUNetv2_extract_fingerprint -d 120 -np 16 

# nnUNetv2_plan_experiment -d 120 -pl nnUNetPlannerResEncM
# nnUNetv2_train 120 3d_fullres_video_mae_vit_decoder_epoch1000_iter350_nomirror_video_pre 0 -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE_NoMirroring -pretrained_weights=/usr/bmicnas02/data-biwi-01/lung_detection/nnDetection_Custom/checkpoints/checkpoint-last_16_224_224.pth

nnUNetv2_train 120 3d_fullres_video_mae_vit_decoder_epoch1000_iter350_nomirror_video_pre 0 -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE_NoMirroring --c

