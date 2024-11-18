# Directory containing the .tar.gz files
SOURCE_DIR="."
# Directory where all extracted files will be merged
DEST_DIR="uncompressed"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through each .tar.gz file in the source directory
for file in "$SOURCE_DIR"/*.tar.gz; 
do
  # Extract the .tar.gz file
  tar -xzf "$file" -C "$SOURCE_DIR"

  # Find the name of the extracted directory
  extracted_dir=$(tar -tf "$file" | head -1 | cut -f1 -d"/")

  # Move contents of extracted directory to the destination directory
  mv "$SOURCE_DIR/$extracted_dir"/* "$DEST_DIR"/

  # Remove the empty extracted directory
  rmdir "$SOURCE_DIR/$extracted_dir"
done
