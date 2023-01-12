# %%

import h5py
import ROOT
import numpy as np


def create_root(f, start, end):
    d = {}
    flatten = True
    for c in list(f.keys()):
        shape = f[c].shape
        if len(shape) > 1:
            if not flatten:
                continue
            temp = np.array(f[c][start:end])
            for j in range(shape[1]):
                d[f"{c}_{j}"] = temp[:, j].astype(np.float32)
            flatten = False
            continue
        d[c] = f[c][start:end].astype(np.float32)

    df = ROOT.RDF.MakeNumpyDataFrame(d)

    df.Snapshot("sig_tree", f"../data/h5train/r{start}-{end}.root", d.keys())

# %%

f = h5py.File("../data/train.h5", "r")

l = f[list(f.keys())[0]].shape[0]

step_size = 1_000_000

start = 11_000_000
while start < l:
    print(f"{start = }")
    create_root(f, start, start+step_size)

    start += step_size

