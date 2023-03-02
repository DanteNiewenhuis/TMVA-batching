# %%

import h5py
import ROOT
import numpy as np
from array import *


def create_root_tree(f, start, end):
    file = ROOT.TFile(f"../data/h5train/r{start}-{end}.root", 'RECREATE')
    t = ROOT.TTree('sig_tree', 'sig_tree')
    vars = {}
    for c in list(f.keys()):
        name = str(c)
        shape = f[c].shape
        n = shape[0]
        if (len(shape) == 1):
            x = array('f', [0.0])
            type = name +"/F"
            print(type)
            t.Branch(name,x,type)
            vars[c] = x
        else:
            N = shape[1]
            v = ROOT.std.vector('float')(N*[0.])
            t.Branch(name, v)
            vars[c]= v

    print("loop on data and fill from ",start,"until",end)
    for i in range(start,end):
        for c in list(f.keys()):
            shape = f[c].shape
            if (len(shape) == 1) :
                vars[c][0] = f[c][i]
            else:
                v = ROOT.std.vector('float')(f[c][i,:])
                ROOT.std.copy(v.begin(),v.end(),vars[c].begin())

        t.Fill()
        if (i and i % 1000 == 0) :
            print("filled ",i,"entries")

    t.Write()


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

step_size = 1_000_000
start = 0
end = 1_000_000
for i in range(40):
    create_root_tree(f, start, end)
    start += step_size
    end += step_size