
import uproot
import numpy as np
import tensorflow as tf

import time

from batch_generator import GetTFDatasets


def h52numpy():
    file = uproot.open("../data/Higgs_data_full.root")
    tree = file["test_tree"]
    branches = tree.arrays()
    target = ""

    res = []

    y = []
    for k in tree.keys():
        if k == target:
            y = branches[k].to_numpy()
            continue
        res.append(branches[k].to_numpy())

    data = np.array(res)
    X = data.transpose()

    return X, y

file_name = "../data/Higgs_data_full.root"
tree_name = "test_tree"

batch_size = 1024

X, y = h52numpy()

ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)

wf = open("../uproot_batching.csv", "w")

start = time.time()
i = 1
for item in ds:
    end = time.time()
    wf.write(f"{end-start}\n")
    i += 1

    start = time.time()

wf.close()

chunk_size = 100_000
ds_train, ds_valid = GetTFDatasets(file_name, tree_name, chunk_size,
                           batch_size, validation_split=0, target="Type")


wf = open("../ROOT_batching.csv", "w")

start = time.time()
print("start ROOT batching")
i = 1
for item in ds_train:
    end = time.time()
    wf.write(f"{end-start}\n")
    i += 1

    start = time.time()

wf.close()