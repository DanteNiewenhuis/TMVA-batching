import ROOT
from batch_generator import Generator
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-delay', type=float)
parser.add_argument('-threaded', default=False, action=argparse.BooleanOptionalAction)

args = parser.parse_args()
delay = args.delay
threaded = args.threaded

print(threaded)

main_folder = "../"

tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

x_rdf = ROOT.RDataFrame(tree_name, file_name)

start = time.time()
columns = x_rdf.GetColumnNames()
print(columns)
print(len(columns))

print(f"getting columns took: {time.time() - start}")

# columns = ["weights", "fjet_C2", "labels"]
filters = []

chunk_rows = 200_000
batch_rows = 5

generator = Generator(file_name, tree_name, chunk_rows, batch_rows, target="Type")

timings = [0]

start = time.time()
last_time = time.time()
for i, batch in enumerate(generator):
    print(batch)

    break

    time.sleep(delay)

    current_time = time.time()
    timings.append(current_time - start)
    
    last_time = current_time


timings = np.array(timings)
print(timings)

diff = timings[1:] - timings[:-1]
print(diff)

# name = "parallel" if threaded else "single"
# with open(f"{main_folder}results/Parallel/python_{name}_{delay*1_000_000:.0f}.csv", "w") as wf:
#     for timing in timings:
#         wf.write(f"{timing}\n")