#!/bin/bash
#!/bin/bash
#SBATCH --job-name=crop_seg_hiera_conv_121_new_label_scratch
#SBATCH --output=sbatch_log/crop_seg_hiera_conv_121_new_label_scratch_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=octopus04
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=bmicgpu07,bmicgpu08,bmicgpu09,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
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



# nnUNetv2_train 888 3d_fullres_torchres_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans_torchres
# nnUNetv2_train 888 3d_fullres_video_mae_vit_decoder_epoch1500_iter350_nomirror all -p VideoMAELPlans -tr nnUNetTrainer_VideoMAE
cd /scratch_net/schusch/qimaqi/cbct_proj/CBCT/nnUNet_bmic

# nnUNetv2_train 888 3d_fullres_hiera_conv_decoder_epoch200_iter350_nomirror 0 -p HieraPlans -tr nnUNetTrainer_VideoHiera_NoMirroring --c

nnUNetv2_train 121 3d_fullres_hiera_conv_decoder_epoch100_iter350_nomirror_scratch 0 -p HieraPlans  -tr nnUNetTrainer_VideoHiera_NoMirroring_finetune100 -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/mae_hiera_large_16x224.pth




#  -pretrained_weights /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_results/Dataset888_teeth/nnUNetTrainer_VideoHiera_NoMirroring__HieraPlans__3d_fullres_hiera_conv_decoder_epoch200_iter350_nomirror/fold_0/checkpoint_best.pth 
