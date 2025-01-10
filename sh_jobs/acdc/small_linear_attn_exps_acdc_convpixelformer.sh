#!/bin/bash
#SBATCH --job-name=nnunet_baseline
#SBATCH --output=sbatch_log/convformer0_2layer_acdc_pos0_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=staff
#SBATCH --nodelist=octopus03
#SBATCH --cpus-per-task=4
#SBATCH --mem 64GB


# Load any necessary modules
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:2 --constraint='titan_xp' --pty bash -i
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --constraint='titan_xp' --pty bash -i
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1   --pty bash -i 
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --nodelist=octopus04 --pty bash -i 

# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --constraint='a6000' --pty bash -i
# srun  --account=staff --cpus-per-task=4 --time 120 --gres=gpu:1  --nodelist=octopus03 --pty bash -i

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

# amos
# nnUNetv2_extract_fingerprint -d 218 -np 16 
# nnUNetv2_extract_fingerprint -d 219 -np 16 
# nnUNetv2_plan_experiment -d 218 -pl nnUNetPlannerResEncM --clean   nnUNetResEncUNetMPlans.json
# nnUNetv2_plan_experiment -d 219 -pl nnUNetPlannerResEncM  nnUNetResEncUNetMPlans.json
# nnUNetv2_plan_experiment -d 218 
# nnUNetv2_plan_experiment -d 27
# nnUNetv2_preprocess -d 219 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans -np 16
# nnUNetv2_preprocess -d 218 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans 

# nnUNetv2_preprocess -d 219 -c 3d_fullres_video_mae -plans_name VideoMAELPlans 

export CUDA_LAUNCH_BLOCKING=1.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_USE_TRT=0 

cd ..
# nnUNetv2_train 27 3d_fullres_convformer_2layer_global_pos_0 0 -p ConvPixelFormerSPlans -tr nnUNetTrainer_ConvPixelPixelFormer

nnUNetv2_train 27 3d_fullres_convformer_2layer_global_pos_0 1 -p ConvPixelFormerSPlans -tr nnUNetTrainer_ConvPixelPixelFormer --c
nnUNetv2_train 27 3d_fullres_convformer_2layer_global_pos_0 2 -p ConvPixelFormerSPlans -tr nnUNetTrainer_ConvPixelPixelFormer
nnUNetv2_train 27 3d_fullres_convformer_2layer_global_pos_0 3 -p ConvPixelFormerSPlans -tr nnUNetTrainer_ConvPixelPixelFormer
nnUNetv2_train 27 3d_fullres_convformer_2layer_global_pos_0 4 -p ConvPixelFormerSPlans -tr nnUNetTrainer_ConvPixelPixelFormer




# nnUNetv2_train 27 3d_fullres_convformer_2layer_0 0 -p ConvPixelFormerSPlans -tr nnUNetTrainer_ConvPixelPixelFormer




# nnUNetv2_train 27 3d_fullres 0 -p nnUNetPlans -tr nnUNetTrainer
