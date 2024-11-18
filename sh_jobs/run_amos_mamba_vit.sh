#!/bin/bash
#!/bin/bash
#SBATCH --job-name=nnunet_baseline
#SBATCH --output=sbatch_log/running_amos_mamba_fold2_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=bmicgpu09
#SBATCH --cpus-per-task=4 
#SBATCH --mem 32GB

### SBATCH --account=staff 
### SBATCH --gres=gpu:5
### SBATCH --constraint='titan_xp'

# Load any necessary modules
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:2 --constraint='titan_xp' --pty bash -i
# srun --account=staff --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --constraint='titan_xp' --pty bash -i
# srun  --cpus-per-task=4 --mem 32GB --time 120 --gres=gpu:1 --pty bash -i


source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate mamba

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
# nnUNetv2_preprocess -d 219 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans -np 16
# nnUNetv2_preprocess -d 218 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans 

# nnUNetv2_preprocess -d 219 -c 3d_fullres_video_mae -plans_name VideoMAELPlans 

export CUDA_LAUNCH_BLOCKING=1.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_USE_TRT=0 

# nnUNetv2_train 219 3d_fullres_linear 0 -p nnUNetResEncLinAttnUNetMPlans 

# nnUNetv2_train 219 3d_fullres_video_mamba_vit_decoder 0 -p MambaPlans  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/videomamba_m16_k400_mask_pt_f8_res224.pth

# nnUNetv2_train 219 3d_fullres_video_mamba_vit_decoder 1 -p MambaPlans  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/videomamba_m16_k400_mask_pt_f8_res224.pth

nnUNetv2_train 219 3d_fullres_video_mamba_vit_decoder 2 -p MambaPlans  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/videomamba_m16_k400_mask_pt_f8_res224.pth





# nnUNetv2_train 219 3d_fullres 0 -p nnUNetResEncUNetMPlans





# nnUNetv2_train 219 3d_fullres_video_mae_vit_decoder 0 -p VideoMAELPlans
#  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/mae_pretrain_vit_large_k700.pth --c


# nnUNetv2_train 219 3d_fullres_video_mae_vit_decoder 0 -p VideoMAELPlans  --c
# 0.8844637576088911
# nnUNetv2_train 219 3d_fullres_video_mae_vit_decoder 1 -p VideoMAELPlans  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/mae_pretrain_vit_large_k700.pth
#  0.883548389623125

# nnUNetv2_train 219 3d_fullres_video_mae_vit_scratch_decoder 0 -p VideoMAELPlans  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/mae_pretrain_vit_large_k700.pth

# nnUNetv2_train 219 3d_fullres_video_mae_vit_scratch_decoder 0 -p VideoMAELPlans --c

# nnUNetv2_train 219 3d_fullres_video_mae_vit_decoder_amos_pretrain 0 -p VideoMAELPlans  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/nnUNet_bmic/checkpoints/checkpoint-00400.pth

# nnUNetv2_train 219 3d_fullres_video_mae_vit_decoder_amos_pretrain 1 -p VideoMAELPlans  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/nnUNet_bmic/checkpoints/checkpoint-00400.pth


#  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/mae_pretrain_vit_large_k700.pth