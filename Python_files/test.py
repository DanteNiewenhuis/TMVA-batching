import ROOT
from batch_generator_parralel import Generator
import time
import numpy as np

main_folder = "../"

tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

x_rdf = ROOT.RDataFrame(tree_name, file_name)

start = time.time()
columns = x_rdf.GetColumnNames()
print(columns)
print(f"getting columns took: {time.time() - start}")

# columns = ["weights", "fjet_C2", "labels"]
filters = []

chunk_rows = 2_000_000
batch_rows = 20_000

start = time.time()
generator = Generator(file_name, tree_name, columns, filters, chunk_rows, batch_rows, target="Type")
end = time.time()

print(f"creating generator took {end - start}")

timings = [0]

delay = 0.01
start = time.time()
last_time = time.time()
for i, batch in enumerate(generator):
    if i % 10 == 0:
        print("TENNNNN")

    current_time = time.time()
    print(f"Getting batch took: {current_time - last_time}")
    timings.append(current_time - start)
    
    last_time = current_time


    if i == 100:
        break


timings = np.array(timings)
print(timings)

diff = timings[1:] - timings[:-1]
print(diff)

# with open(f"{main_folder}results/Parralel/parralel.csv", "w") as wf:
#     for timing in timings:
#         wf.write(f"{timing}\n")