#!/bin/bash
#SBATCH --job-name=n120_3d_fullres
#SBATCH --output=sbatch_log/n122_3d_fullres_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu07,bmicgpu08,bmicgpu09,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 64GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.eins.qi@gmail.com
### SBATCH --account=staff 
### SBATCH --gres=gpu:5

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
# nnUNetv2_extract_fingerprint -d 122 -np 16 

# nnUNetv2_plan_experiment -d 122 -pl nnUNetPlannerResEncM


# nnUNetv2_preprocess -d 122 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans -np 16
nnUNetv2_train 122 3d_fullres 0 -p nnUNetResEncUNetMPlans  -tr nnUNetTrainerNoMirroring
