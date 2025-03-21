#!/bin/bash
#!/bin/bash
#SBATCH --job-name=crop_seg
#SBATCH --output=sbatch_log/align_teeth_mae_teeth_pre_acc_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=bmicgpu08
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
### SBATCH --mem-per-cpu=16GB

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


cd /scratch_net/schusch/qimaqi/cbct_proj/CBCT/nnUNet_bmic
# 3d_fullres_video_mae_vit_decoder_epoch1500_iter350_nomirror    3d_fullres_video_mae_vit_decoder_epoch1500_iter350_nomirror
# nnUNetv2_train 888 3d_fullres_torchres_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans_torchres

# start 
# nnUNetv2_train 888 3d_fullres_video_mae_vit_decoder_epoch1500_iter350_nomirror_pre all -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE_NoMirroring -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/mae_pretrain_vit_large_k700.pth


# resume
# nnUNetv2_train 888 3d_fullres_video_mae_vit_decoder_epoch1500_iter350_nomirror_teeth_pre 0 -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE_NoMirroring  -pretrained_weights=/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/align_model/align_teeth_pretrain/checkpoint-00999.pth


# nnUNetv2_train 888 3d_fullres_video_mae_vit_decoder_epoch1500_iter350_nomirror_teeth_pre 0 -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE_NoMirroring --c

# nnUNetv2_train 888 3d_fullres_video_mae_vit_decoder_epoch1000_iter350_nomirror_teeth_pre 0 -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE_NoMirroring --c


# nnUNetv2_train 888 3d_fullres_video_mae_vit_decoder_epoch1500_iter350_nomirror_teeth_pre_acc 0 -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE_NoMirroring  -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/nnUNet_bmic/checkpoints/teeth_only/teeth_only.pth


nnUNetv2_train 888 3d_fullres_video_mae_vit_decoder_epoch1500_iter350_nomirror_teeth_pre_acc 0 -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE_NoMirroring  --c