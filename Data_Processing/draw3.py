# Function to determine the data range for each column and perform mapping
import os

def process_pdb_files(folder_path):
    # Check if the target folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None

    # Set to store the data in the 4th column of all files
    result_set = set()

    # Get the list of all PDB files in the folder
    pdb_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".pdb")]

    if not pdb_files:
        print(f"Error: No PDB files found in the specified folder.")
        return None

    # Iterate through each PDB file
    for pdb_file in pdb_files:
        file_path = os.path.join(folder_path, pdb_file)
        with open(file_path, 'r') as file:
            # Iterate through each line of the file
            for line in file:
                # Process only lines starting with 'ATOM'
                if line.startswith("ATOM"):

                    # # Use split() to separate the line and get the data in the 12th column
                    # columns = line.split()
                    # if len(columns) >= 12:
                    #     data_in_column_12 = columns[11].strip()
                    #     result_set.add(data_in_column_12)
                    
                    # # Use split() to separate the line and get the data in the 3rd column
                    # columns = line.split()
                    # if len(columns) >= 3:
                    #     data_in_column_3 = columns[2].strip()
                    #     result_set.add(data_in_column_3)
                    
                    # Use split() to separate the line and get the data in the 4th column
                    columns = line.split()
                    if len(columns) >= 4:
                        data_in_column_4 = columns[3].strip()
                        result_set.add(data_in_column_4)

    # Convert the set to a tuple and output
    result_tuple = tuple(result_set)
    print("Result Set:", sorted(result_tuple))

    return result_tuple

# Specify the folder path
folder_path = ""

# Call the function to process PDB files
process_pdb_files(folder_path)


'''
36 items: atom_type_mapping = {'C': 1, 'CA': 2, 'CB': 3, 'CD': 4, 'CD1': 5, 'CD2': 6, 'CE': 7, 'CE1': 8, 'CE2': 9, 'CE3': 10, 'CG': 11, 'CG1': 12, 'CG2': 13, 'CH2': 14, 'CZ': 15, 'CZ2': 16, 'CZ3': 17, 'N': 18, 'ND1': 19, 'ND2': 20, 'NE': 21, 'NE1': 22, 'NE2': 23, 'NH1': 24, 'NH2': 25, 'NZ': 26, 'O': 27, 'OD1': 28, 'OD2': 29, 'OE1': 30, 'OE2': 31, 'OG': 32, 'OG1': 33, 'OH': 34, 'SD': 35, 'SG': 36}
20 items: residue_type_mapping = {'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6, 'GLU': 7, 'GLY': 8, 'HIS': 9, 'ILE': 10, 'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14, 'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20}
4 items: element_mapping = {'C': 1, 'N': 2, 'O': 3, 'S': 4}
'''