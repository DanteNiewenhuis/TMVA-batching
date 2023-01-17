import os

file_list = os.listdir("../data/h5train")

s = "hadd ../data/h5train_big_combined.root"

for file in file_list:
    s += f" ../data/h5train/{file} "

os.system(s)