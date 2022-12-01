import os

file_list = os.listdir("../data/h5data")

s = "hadd ../data/combined.root"

for file in file_list:
    s += f" ../data/h5data/{file} "

os.system(s)