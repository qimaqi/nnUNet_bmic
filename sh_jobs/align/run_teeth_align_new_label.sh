#!/bin/bash
#!/bin/bash
#SBATCH --job-name=crop_seg
#SBATCH --output=sbatch_log/align_teeth_mae_nnunet_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=bmicgpu06
#SBATCH --cpus-per-task=4
#SBATCH --mem 120GB
##SBATCH --mem-per-cpu=16GB

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


# nnUNetv2_extract_fingerprint -d 120 -np 16 
# nnUNetv2_extract_fingerprint -d 121 -np 16 
# nnUNetv2_plan_experiment -d 120 -pl nnUNetPlannerResEncL  nnUNetResEncUNetLPlans.json
# nnUNetv2_plan_experiment -d 121 -pl nnUNetPlannerResEncL  nnUNetResEncUNetLPlans.json
nnUNetv2_preprocess -d 120 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans -np 4
nnUNetv2_preprocess -d 121 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans -np 4



# nnUNetv2_plan_experiment -d 218 -pl nnUNetPlannerResEncM --clean   nnUNetResEncUNetMPlans.json

# nnUNetv2_plan_experiment -d 218 
# nnUNetv2_plan_experiment -d 27
# nnUNetv2_preprocess -d 219 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans -np 16
# nnUNetv2_preprocess -d 218 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans 



# step1: check data label of align
# seg_folder = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_preprocessed/Dataset888_teeth/nnUNetPlans_3d_fullres 

# cd /scratch_net/schusch/qimaqi/cbct_proj/CBCT/nnUNet_bmic/scripts

# python check_align_label.py
# check how to make new model final output layer based on old model

# nnUNetv2_train 888 3d_fullres 0 -p nnUNetResEncUNetLPlans  -tr nnUNetTrainerNoMirroring --c

# nnUNetv2_train 121 3d_fullres 0 -p nnUNetResEncUNetLPlans  -tr nnUNetTrainerNoMirroring --c


# 
