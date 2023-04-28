import os


main_folder = "/home/danteniewenhuis/Documents/CERN/TMVA-batching"
folder = f"{main_folder}/data/file_sizes_bench"

def create_file(num):
    s = f"hadd {folder}/{num}.root " + (f"{folder}/base.root " * num)

    os.system(s)

for i in range(21, 51):
    create_file(i)