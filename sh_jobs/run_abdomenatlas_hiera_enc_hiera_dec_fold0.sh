#!/bin/bash
#!/bin/bash
#SBATCH --job-name=nnunet_baseline
#SBATCH --output=sbatch_log/abdomaltas_hiera_hiear_0_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=bmicgpu07
#SBATCH --cpus-per-task=4 
#SBATCH --mem 96GB

### SBATCH --account=staff 
### SBATCH --gres=gpu:5
### SBATCH --constraint='titan_xp'

# Load any necessary modules
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:2 --constraint='titan_xp' --pty bash -i
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --constraint='titan_xp' --pty bash -i
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --pty bash -i


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
# nnUNetv2_extract_fingerprint -d 224 
# nnUNetv2_plan_experiment -d 224 -pl nnUNetPlannerResEncL_torchres
# nnUNetv2_preprocess -d 224 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans_torchres -np 2


export CUDA_LAUNCH_BLOCKING=1.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_USE_TRT=0 

# nnUNetv2_train 224 3d_fullres_hiera_hiera_decoder_epoch1500_iter500 0 -p HieraPlans -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/mae_hiera_large_16x224.pth -tr nnUNetTrainer_VideoHiera

nnUNetv2_train 224 3d_fullres_hiera_hiera_decoder_epoch1500_iter500 0 -p HieraPlans  -tr nnUNetTrainer_VideoHiera  --c


# nnUNetv2_train 224 3d_fullres_video_mae_conv_decoder_e1500 all -p VideoMAEPlans --c


# # -num_gpus 1

# nnUNetv2_train 224 3d_fullres 0 -p nnUNetResEncUNetLPlans_torchres 
