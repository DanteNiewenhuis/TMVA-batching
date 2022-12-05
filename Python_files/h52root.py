# %%

import h5py
import ROOT
import numpy as np

def create_root(f, start, end):
    d = {}
    for c in list(f.keys()):
        shape = f[c].shape
        if len(shape) > 1:
            continue
        d[c] = f[c][start:end]  

    df = ROOT.RDF.MakeNumpyDataFrame(d)

    df.Snapshot("sig_tree", f"../data/h5train/r{start}-{end}.root", d.keys())

# %%

f = h5py.File("../data/train.h5", "r")

l = f[list(f.keys())[0]].shape[0]


start = 0

step_size = 1_000_000

while start < l:
    print(f"{start = }")
    create_root(f, start, start+step_size)

    start += step_size
