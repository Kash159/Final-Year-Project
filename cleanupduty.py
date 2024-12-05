import os

# Paths to your Normal and TB folders
normal_folder = "/Users/kashyapkaliyur/Desktop/ML Projects/Final Year Project/Shenzen Dataset/images/Normal"
tb_folder = "/Users/kashyapkaliyur/Desktop/ML Projects/Final Year Project/Shenzen Dataset/images/TB"

# List of folders to clean up
folders_to_clean = [normal_folder, tb_folder]

# Loop through each folder and remove files ending with "_mask.png"
for folder in folders_to_clean:
    for filename in os.listdir(folder):
        if filename.endswith("_mask.png"):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

print("Cleanup complete.")
