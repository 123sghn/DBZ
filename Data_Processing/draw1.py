# Function to extract pdb files containing "model_1" from the original folder
import os
import shutil

def extract_model_1_pdb_files(source_folder, destination_folder):
    # Ensure the destination folder exists, create it if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get the list of all files in the source folder
    file_list = os.listdir(source_folder)

    # Iterate through the file list, find files containing "model_1" and 'Positive', and copy them to the destination folder
    for filename in file_list:
        if "model_1" in filename and 'Positive' in filename and filename.endswith(".pdb"):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            shutil.copyfile(source_path, destination_path)
            print(f"File '{filename}' copied to '{destination_folder}'.")

# Specify the source folder and destination folder
source_directory = ""
destination_directory = ""

# Call the function to extract files
extract_model_1_pdb_files(source_directory, destination_directory)