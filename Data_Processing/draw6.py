# Concatenate two CSV files to obtain 3D+BINA data
import csv
import os

def concatenate_csv(input_file1, input_file2, output_file):
    # Read the first CSV file
    with open(input_file1, 'r') as file1:
        reader1 = csv.reader(file1)
        data1 = [row for row in reader1]

    # Read the second CSV file
    with open(input_file2, 'r') as file2:
        reader2 = csv.reader(file2)
        data2 = [row for row in reader2]

    # Ensure that the number of rows in the two files is the same
    if len(data1) != len(data2):
        print("Error: The number of rows in the two files is different.")
        print("The length of 1 is " , len(data1))
        print('The length of 2 is ' , len(data2))
        return
    
    # Get the output file directory
    output_dir = os.path.dirname(output_file)

    # If the directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Concatenate each row of the two files and write to a new CSV file
    with open(output_file, 'w', newline='') as output:
        writer = csv.writer(output)
        for row1, row2 in zip(data1, data2):
            new_row = row1 + row2
            writer.writerow(new_row)

temp_t = ['BS','CG','EC','GK','MT','ST']

for i in range(len(temp_t)):
    # Example usage
    concatenate_csv('', '', '')