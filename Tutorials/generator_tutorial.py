import time
import numpy as np
import ROOT

ROOT.EnableThreadSafety()

main_folder = "/home/dante-niewenhuis/Documents/TMVA-batching"

tree_name = "test_tree"
file_name = f"{main_folder}/data/Higgs_data_full.root"
# file_name = f"{main_folder}/data/hvector.root"
# file_name = f"{main_folder}/data/simple_data.root"


chunk_size = 1_000_000
batch_size = 1024


max_vec_sizes = {"f1": 2, "f4": 3}

ds_train, ds_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    tree_name,
    file_name,
    batch_size,
    chunk_size,
    validation_split=0.3,
    max_vec_sizes=max_vec_sizes,
)

print(f"{ds_train.columns = }")


for i, b in enumerate(ds_train):
    print(f"Training batch {i} => {b.shape = }")
    break

# ds_train.DeActivate()
# print(f"Starting Validation")
# for i, b in enumerate(ds_validation):
#     print(f"Validation batch {i} => {len(b) = }")
#     break

print("END OF FILE")
