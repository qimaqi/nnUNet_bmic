#!/bin/bash
#SBATCH --job-name=nnunet_baseline
#SBATCH --output=sbatch_log/plain_unet_3layer_amos_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu06
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
##SBATCH --mem 32GB

##SBATCH --account=staff 
##SBATCH --gres=gpu:1
##SBATCH --constraint='titan_xp'

# Load any necessary modules
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:2 --constraint='titan_xp' --pty bash -i
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --constraint='titan_xp' --pty bash -i
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1   --pty bash -i 
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --nodelist=bmicgpu06 --pty bash -i 


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

# acdc
# convert
cd /scratch_net/schusch/qimaqi/cbct_proj/CBCT/nnUNet_bmic/nnunetv2/dataset_conversion
python Dataset220_KiTS2023.py  /usr/bmicnas02/data-biwi-01/ct_video_mae/ct_datasets/KiT23/kits23_data
# nnUNetv2_extract_fingerprint -d 218 -np 16 
# nnUNetv2_extract_fingerprint -d 27 -np 16 
# # nnUNetv2_plan_experiment -d 218 -pl nnUNetPlannerResEncM --clean   nnUNetResEncUNetMPlans.json
# nnUNetv2_plan_experiment -d 27 -pl nnUNetPlannerResEncM  nnUNetResEncUNetMPlans.json
# # nnUNetv2_plan_experiment -d 218 
# # nnUNetv2_plan_experiment -d 219
# # nnUNetv2_preprocess -d 219 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans -np 16
# nnUNetv2_preprocess -d 27 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans 
