#!/bin/bash
#SBATCH --job-name=align_teeth_continue_training_121_100e_scratch
#SBATCH --output=sbatch_log/align_teeth_continue_training_121_100e_scratch_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu06,bmicgpu07,bmicgpu08,bmicgpu09,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 160GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qi.ma@vision.ee.ethz.ch

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


cd /scratch_net/schusch/qimaqi/cbct_proj/CBCT/nnUNet_bmic

# nnUNetv2_extract_fingerprint -d 121 -np 16 
# nnUNetv2_plan_experiment -d 121 -pl nnUNetPlannerResEncL  nnUNetResEncUNetLPlans.json
# nnUNetv2_preprocess -d 121 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans -np 4


nnUNetv2_train 121 3d_fullres_scratch 0 -p nnUNetResEncUNetLPlans  -tr nnUNetTrainerNoMirroring_finetune100 


# -pretrained_weights /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_results/Dataset888_teeth/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth
