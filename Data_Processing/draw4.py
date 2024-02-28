# Function to extract column data from PDB files and perform padding and concatenation
import os
import csv

def map_to_index(value, mapping_dict):
    return mapping_dict[value] if value in mapping_dict else 0

def process_pdb_files(input_folder, output_csv):
    # Define mapping dictionaries
    atom_type_mapping = {'C': 1, 'CA': 2, 'CB': 3, 'CD': 4, 'CD1': 5, 'CD2': 6, 'CE': 7, 'CE1': 8, 'CE2': 9, 'CE3': 10, 'CG': 11, 'CG1': 12, 'CG2': 13, 'CH2': 14, 'CZ': 15, 'CZ2': 16, 'CZ3': 17, 'N': 18, 'ND1': 19, 'ND2': 20, 'NE': 21, 'NE1': 22, 'NE2': 23, 'NH1': 24, 'NH2': 25, 'NZ': 26, 'O': 27, 'OD1': 28, 'OD2': 29, 'OE1': 30, 'OE2': 31, 'OG': 32, 'OG1': 33, 'OH': 34, 'SD': 35, 'SG': 36}
    residue_type_mapping = {'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6, 'GLU': 7, 'GLY': 8, 'HIS': 9, 'ILE': 10, 'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14, 'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20}
    element_mapping = {'C': 1, 'N': 2, 'O': 3, 'S': 4}

    # Open the CSV file for appending data without including a header
    with open(output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Get the list of all PDB files in the folder and sort them in ascending order by file name
        pdb_files = sorted([filename for filename in os.listdir(input_folder) if filename.endswith(".pdb")], key=lambda x: int(x.split("_")[1]))

        if not pdb_files:
            print(f"Error: No PDB files found in the specified folder.")
            return

        # Iterate through each PDB file
        for pdb_file in pdb_files:
            file_path = os.path.join(input_folder, pdb_file)
            with open(file_path, 'r') as file:
                # Initialize lists for partial data of each line
                atom_type_list, residue_type_list, col_7_list, col_8_list, col_9_list, col_11_list, col_12_list = [], [], [], [], [], [], []

                # Iterate through each line of the file
                for line in file:
                    if line.startswith("ATOM"):
                        # Extract data from the respective columns
                        columns = line.split()
                        atom_type_list.append(map_to_index(columns[2].strip(), atom_type_mapping))
                        residue_type_list.append(map_to_index(columns[3].strip(), residue_type_mapping))
                        col_7_list.append(columns[6].strip())
                        col_8_list.append(columns[7].strip())
                        col_9_list.append(columns[8].strip())
                        col_11_list.append(columns[10].strip())
                        col_12_list.append(map_to_index(columns[11].strip(), element_mapping))

                # Pad the length to 210, filling the gaps with zeros
                atom_type_list.extend([0] * (210 - len(atom_type_list)))
                residue_type_list.extend([0] * (210 - len(residue_type_list)))
                col_7_list.extend(['0'] * (210 - len(col_7_list)))
                col_8_list.extend(['0'] * (210 - len(col_8_list)))
                col_9_list.extend(['0'] * (210 - len(col_9_list)))
                col_11_list.extend(['0'] * (210 - len(col_11_list)))
                col_12_list.extend([0] * (210 - len(col_12_list)))

                # Concatenate all partial data into one row and write it to the CSV file
                csv_row = tuple(atom_type_list + residue_type_list + col_7_list + col_8_list + col_9_list + col_11_list + col_12_list)
                csv_writer.writerow(csv_row)

# Specify the input folder path and the output CSV file path
input_folder_path = ""
output_csv_path = ""

# Remove the output file to avoid appending when running the script again
if os.path.exists(output_csv_path):
    os.remove(output_csv_path)

# Call the function to process PDB files
process_pdb_files(input_folder_path, output_csv_path)
