#!/bin/bash
# Source and target directories
# SOURCE_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/imagesTr"
# TARGET_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset122_ToothFairy2_and_Align/imagesTr/"
# SOURCE_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/labelsTr"
# TARGET_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset122_ToothFairy2_and_Align/labelsTr/"
# SOURCE_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset120_ToothFairy2_to_Align/labelsTr"
# TARGET_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset122_ToothFairy2_and_Align/labelsTr"


# SOURCE_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset121_ToothFairy2_to_Align_new_labels/imagesTr"
# TARGET_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset123_ToothFairy2_and_Align_new_labels
# /imagesTr"
# SOURCE_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset121_ToothFairy2_to_Align_new_labels/labelsTr"
# TARGET_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset123_ToothFairy2_and_Align_new_labels
# /labelsTr"

# SOURCE_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/imagesTr"
# TARGET_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset123_ToothFairy2_and_Align_new_labels
# /imagesTr"

SOURCE_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset888_teeth/labelsTr"
TARGET_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset123_ToothFairy2_and_Align_new_labels
/labelsTr"


# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory does not exist: $SOURCE_DIR"
  exit 1
fi

# Check if target directory exists, create it if it doesn't
if [ ! -d "$TARGET_DIR" ]; then
  echo "Target directory does not exist. Creating: $TARGET_DIR"
  mkdir -p "$TARGET_DIR"
fi

# Loop through all files ending with .nii.gz in the source directory
for file in "$SOURCE_DIR"/*.nii.gz; do
  # Check if the file exists (in case no files match the pattern)
  if [ -e "$file" ]; then
    # Get the base name of the file
    filename=$(basename "$file")
    # Create a symbolic link in the target directory
    ln -s "$file" "$TARGET_DIR/$filename"
    echo "Created symlink for: $filename"
  else
    echo "No .nii.gz files found in the source directory."
    break
  fi
done

echo "All .nii.gz files have been symlinked to the target directory."