import os

# Define the path to the mask images
mask_dir = "Mask-images"

# Loop through all files in the mask directory
for filename in os.listdir(mask_dir):
    # Check if the file ends with "_mask.png"
    if filename.endswith("_mask.png"):
        # Generate the new filename by removing "_mask" from the filename
        new_filename = filename.replace("_mask", "")
        
        # Get the full paths for the old and new filenames
        old_path = os.path.join(mask_dir, filename)
        new_path = os.path.join(mask_dir, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")
