import ROOT
ROOT.EnableThreadSafety()

# from BatchTimer import BatchTimer

import numpy as np
import time
import argparse

# import tensorflow as tf

main_folder = "/home/dante-niewenhuis/Documents/TMVA-batching"

tree_name = "test_tree"
file_name = f"{main_folder}/data/Higgs_data_full.root"

batch_rows = 10
chunk_rows = 100

ds_train, ds_validation = ROOT.TMVA.Experimental.GetTFDatasets(file_name, tree_name, chunk_rows,
                           batch_rows, target="Type", validation_split=0.3, max_chunks=2)

for chunk in ds_train:
    print(chunk)


time.sleep(1)    