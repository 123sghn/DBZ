# Function to find the maximum number of lines in all PDB files to determine the padding length
import os

def find_max_lines_pdb_file(folder_path):
    # Check if the target folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None

    # Get the list of all PDB files in the folder
    pdb_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".pdb")]

    if not pdb_files:
        print("Error: No PDB files found in the specified folder.")
        return None

    max_lines = 0
    max_lines_file = None

    # Iterate through PDB files to find the file with the maximum number of lines
    for pdb_file in pdb_files:
        file_path = os.path.join(folder_path, pdb_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            current_lines = len(lines)
            if current_lines > max_lines:
                max_lines = current_lines
                max_lines_file = pdb_file

    if max_lines_file is not None:
        print(f"The PDB file with the maximum number of lines is '{max_lines_file}' with {max_lines-4} lines.")
        return max_lines
    else:
        print("Error: Unable to determine the PDB file with the maximum number of lines.")
        return None

# Specify the folder path
folder_path = ""

# Call the function to find the PDB file with the maximum number of lines
max_lines = find_max_lines_pdb_file(folder_path)

# Since the maximum length is determined to be 207, pad it to 210
# The PDB file with the maximum number of lines is 'Positive_2028_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb' with 207 lines.
