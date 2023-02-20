import ROOT
from batch_generator import GetGenerators
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

chunk_rows = 2_000
batch_rows = 128

train_generator, test_generator = GetGenerators(file_name, tree_name, chunk_rows,
                           batch_rows, target="Type", weights="missing_energy_phi", train_ratio=0.7)


# timings = [0]

# start = time.time()
# last_time = time.time()
for i, batch in enumerate(train_generator):
    print(f"train batch {i} => {batch[0].shape, batch[1].shape, batch[2].shape}")

    break


# for i, batch in enumerate(test_generator):
#     print(f"test batch {i} => {batch[0].shape}")
