import ROOT
from batch_generator import BatchGenerator
import time
import numpy as np
import argparse

main_folder = "../"

tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

x_rdf = ROOT.RDataFrame(tree_name, file_name)

start = time.time()
columns = x_rdf.GetColumnNames()

filters = []

chunk_rows = 10
batch_rows = 10

generator = BatchGenerator(file_name, tree_name, chunk_rows,
                           batch_rows, target="Type", weights="missing_energy_phi")

timings = [0]

start = time.time()
last_time = time.time()
for i, batch in enumerate(generator):
    print(f"batch {i} => {batch[0].shape}, {batch[1]}, {batch[2]}")

    break
