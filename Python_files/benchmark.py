from batch_generator import Generator
import ROOT
import numpy
import time

import matplotlib.pyplot as plt

main_folder = "../"
file_name = f"{main_folder}data/r0-10.root"
tree_name = "sig_tree"


columns = ["m_jj", "m_jjj", "m_jlv"] 
filters = ["m_jj < 0.9"]
x_rdf = ROOT.RDataFrame(tree_name, file_name)

columns = x_rdf.GetColumnNames()

num_columns = len(columns)
batch_rows = 1
chunk_rows = 2

start = time.time()
generator = Generator(file_name, tree_name, columns, filters, chunk_rows, batch_rows, use_whole_file=True)

middle = time.time()

print(f"loading took: {middle - start}")

times = [0]

for batch in generator:
    print(batch)

    times.append(time.time() - middle)

# for i, batch in enumerate(generator):
#     times.append(time.time() - middle)

#     if (i % 50 == 0):
#         print(f"batch {i}")


end = time.time()
print(f"batching took: {end - middle}")

with open(f"{main_folder}results/python_data.txt", "w") as f:
    f.write(",".join([str(i) for i in times]))