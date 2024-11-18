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



# toothfairy
# nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
# nnUNetv2_extract_fingerprint -d 119 -np 16 
# nnUNetv2_plan_experiment -d 119 -np 16 

# nnUNetv2_train 119 3d_fullres all -p nnUNetResEncUNetLPlans_torchres
# nnUNetv2_train 119 3d_fullres_video_mae_vit_decoder 0 -p VideoMAELPlans -pretrained_weights=/scratch_net/schusch/qimaqi/cbct_proj/CBCT/Video_MAE_Seg/checkpoints/mae_pretrain_vit_large_k700.pth 

# nnUNetv2_plan_experiment -d 888 -pl nnUNetPlannerResEncL_torchres
cd ..
# nnUNetv2_train 888 3d_fullres_torchres_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans_torchres
nnUNetv2_train 888 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_onlyMirror01_1500ep
# nnUNet_results=${nnUNet_results}_2 nnUNetv2_train 119 3d_fullres_torchres_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_onlyMirror01_1500ep

# nnUNetv2_plan_experiment -d 119 -pl nnUNetPlannerResEncL_torchres
# nnUNetv2_preprocess -d 119 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans_torchres -np 16
# nnUNetv2_preprocess -d 888 -c 3d_fullres_torchres_ps160x320x320_bs2 -plans_name nnUNetResEncUNetLPlans_torchres -np 8
# nnUNetv2_train 119 3d_fullres all -p nnUNetResEncUNetLPlans_torchres


# # align one
# nnUNetv2_extract_fingerprint -d 888 -np 16 
# nnUNetv2_plan_experiment -d 888 -pl nnUNetPlannerResEncL_torchres
# nnUNetv2_preprocess -d 888 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans_torchres -np 16

# nnUNetv2_train 888 3d_fullres 0 -p nnUNetResEncUNetLPlans_torchres --c
# nnUNetv2_preprocess -d 888 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans_torchres 

# nnUNetv2_train 888 3d_fullres_torchres_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_onlyMirror01_1500ep
# nnUNetv2_plan_experiment -d 888 -pl nnUNetPlannerResEncM
# nnUNetv2_preprocess -d 888 -c 3d_fullres -plans_name nnUNetResEncUNetMPlans 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 --continue_prediction



# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/0031505282 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# /teeth_datasets/Test/nnunet_eval/rename_data/

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/0129103121 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/0187169607 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/0202371374 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/0539614900 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# problem with 1295161953
# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/1295161953 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/1310284057 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/1638221111 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/1710701244 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# problem with 2-1271
# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/2-1271 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/2-1310 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 


# problem with 2-1483
# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/2-1483 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 4

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/2-1530 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/2-1690 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 


# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/2-1726 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 


# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/3331357025 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 


# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/3892814773 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/4402296261 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/44251 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/442510 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/442511 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/44253 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/44255 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/44258 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/4576056308 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 


# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/4710655036 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/5012048974 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 

# nnUNetv2_predict -i /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/rename_data/50131916 -o /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/teeth_datasets/Test/nnunet_eval/nnunet_pred -d 888 -p nnUNetResEncUNetLPlans_torchres -f all -c 3d_fullres -npp 1 -nps 1 










