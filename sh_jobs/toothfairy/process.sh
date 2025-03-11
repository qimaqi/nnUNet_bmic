#!/bin/bash
#!/bin/bash
#SBATCH --job-name=crop_seg
#SBATCH --output=sbatch_log/teeth_nnunet_res_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=bmicgpu07
#SBATCH --cpus-per-task=4 
#SBATCH --mem 32GB

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




# nnUNetv2_extract_fingerprint -d 120 -np 16 

# nnUNetv2_plan_experiment -d 120 -pl nnUNetPlannerResEncL

# nnUNetv2_preprocess -d 120 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans 

# nnUNetv2_extract_fingerprint -d 121 -np 16 

# nnUNetv2_plan_experiment -d 121 -pl nnUNetPlannerResEncL

nnUNetv2_preprocess -d 121 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans 
