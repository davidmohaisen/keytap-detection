"""
A scipt for dividing recording folders into different folders according to the string start and end indexes.
"""
import glob
import os

# input: filepath
# output: list of strings, each delimited with tab character in the input file
def read_data_from_tab_delimited_file(path):
    with open(path) as f:
        line = f.readline()
        line = line.replace("\n","").split("\t")
        return line
# Writes the given string list by putting a tab delimeter between them to the given file.
def write_data_to_file(data, filename):
    with open(filename, 'w+') as f:
        f.write('\t'.join(data))

# Extracts 1 record.
def extractRecord(input_folder, indexes, output_folder):
    files = glob.glob(input_folder)
    start, end = indexes
    for f in files:
        # If the file is not the file containing keys pressed and the labels, divide that file.
        if ('right' in f) or ('left' in f) or ('timestamp' in f):
            # get the filename
            name = f.split('\\')[-1]
            data_string_list = read_data_from_tab_delimited_file(f)
            data_to_split = data_string_list[start:end]
            write_data_to_file(data_to_split, output_folder+name)

# Extracts multiple records.
# For each tuple in the indexes list, creates a folder and splits each file in the input folder and puts in the corresponding newly created folder.
def extractRecords(input_folder, indexes, output_folder):
    files = glob.glob(input_folder)
    for f in files:
        # If the file is not the file containing keys pressed and the labels, divide that file.
        if ('right' in f) or ('left' in f) or ('timestamp' in f):
            # get the filename
            name = f.split('\\')[-1]
            # open the file to split
            data_string_list = read_data_from_tab_delimited_file(f)
            # for each split...
            for i, (start, end) in enumerate(indexes):
                # construct the filename
                folder_name = output_folder + '\\' + str(i)
                
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                output_filename = folder_name + '\\' + name
                
                # get the corresponding split
                data_split = data_string_list[start:end]
                # write the split to the folder
                write_data_to_file(data_split, output_filename)


# extractRecord(".\\record\\*", (100, 200), ".\\record_1\\")

# indexes = [(0,5),(15,25)]
# extractRecords(".\\record\\*", indexes, ".\\record_1\\")