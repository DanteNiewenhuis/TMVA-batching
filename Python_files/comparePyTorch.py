
import uproot
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time

from batch_generator import GetPyTorchDataLoader


def h52numpy():
    file = uproot.open("../data/Higgs_data_full.root")
    tree = file["test_tree"]
    branches = tree.arrays()
    target = "Type"

    res = []

    y = []
    for k in tree.keys():
        if k == target:
            y = branches[k].to_numpy()
            print(f"{y = }")
            continue
        res.append(branches[k].to_numpy())

    data = np.array(res)
    X = data.transpose()

    return X, y

file_name = "../data/Higgs_data_full.root"
tree_name = "test_tree"

batch_size = 1024

X, y = h52numpy()



class BatchLoader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index: int):
        return self.X[index, :], self.y[index]

ds = BatchLoader(X, y)

loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
wf = open("../uproot_batching_pytorch.csv", "w")

start = time.time()
for item in loader:
    end = time.time()
    wf.write(f"{end-start}\n")

    start = time.time()

wf.close()

chunk_size = 100_000
ds_train, ds_valid = GetPyTorchDataLoader(file_name, tree_name, chunk_size,
                           batch_size, validation_split=0, target="Type")


wf = open("../ROOT_batching_pytorch.csv", "w")

start = time.time()
print("start ROOT batching")
i = 1
for item in ds_train():
    end = time.time()
    wf.write(f"{end-start}\n")
    i += 1

    start = time.time()

wf.close()