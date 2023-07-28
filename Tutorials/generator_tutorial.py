import time
import numpy as np
import ROOT

# ROOT.EnableThreadSafety()

main_folder = "/home/dante/Documents/TMVA-batching"

tree_name = "test_tree"
# file_name = f"{main_folder}/data/Higgs_data_full.root"
# file_name = f"{main_folder}/data/hvector.root"
file_name = f"{main_folder}/data/small_data.root"

# tree_name = "sig_tree"
# file_name = "http://root.cern/files/Higgs_data.root"

chunk_size = 500
batch_size = 5

# filters = ["f1 > 30", "f2 < 70", "f3 == true"]
# filters = ["Type == true"]
# max_vec_sizes = {"f4": 1, "f5": 1, "f6": 1}

target = ["f1", "f2"]
target = "f1"

ds_train, ds_validation = ROOT.TMVA.Experimental.CreateTFDatasets(
    tree_name,
    file_name,
    batch_size,
    chunk_size,
    validation_split=0.3,
    target=target,
    shuffle=False,
)

print(f"{ds_train.columns = }")
print(f"{ds_train.train_columns = }")
print(f"{ds_train.target_columns = }")


for x, y in ds_train:
    print(f"Training batch => {x = }, {y = }")
    break

for i, b in enumerate(ds_validation):
    print(f"Validation batch {i} => {b = }")
    break
