#!/bin/bash
#SBATCH --job-name=crop_seg
#SBATCH --output=sbatch_log/align_teeth_medformer_teeth_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=octopus04
##SBATCH --cpus-per-task=4
#SBATCH --mem 120GB
#£SBATCH --nodelist=bmicgpu06
#£SBATCH --cpus-per-task=4

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

# resume
nnUNetv2_train 888 3d_fullres_medformer_epoch1500_iter350_nomirror 0 -p MedformerPlans -tr nnUNetTrainer_Medformer_NoMirroring --c

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/imagesTs -o /scratch_net/schusch/qimaqi/cbct_proj/CBCT/align_eval/medformer_baseline/pred -d 888 -c 3d_fullres_medformer_epoch1500_iter350_nomirror -p MedformerPlans -f 0 -tr nnUNetTrainer_Medformer_NoMirroring

