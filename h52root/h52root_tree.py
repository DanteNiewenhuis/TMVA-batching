# %%

import h5py
import ROOT
import numpy as np
from array import *

def create_root_tree_fast(f, start, end, bsize) :
    ROOT.gInterpreter.Declare('#include "ConverterRootTree.h"')
    conv = ROOT.ConverterRootTree('tree.root','tree',bsize,21,4)
    for c in list(f.keys()):
        name = str(c)
        shape = f[c].shape
        n = shape[0]
        if (len(shape) == 1):
            conv.AddFloatBranch(name)
        else:
            N = shape[1]
            conv.AddVecBranch(name,N)

    print("loop on data and fill from ",start,"until",end)
    for i in range(start,end,bsize):
        s = []
        v = []
        for c in list(f.keys()):
            shape = f[c].shape
            if (len(shape) == 1):
                s.append(f[c][start:start+bsize])
            else:
                v.append(f[c][start:start+bsize,:])
        xs = np.stack(s,axis=-1)
        xv = np.stack(v,axis=1)
        print(xs.shape,xs.dtype,xv.shape,xv.dtype)
        conv.SetScalarData(xs.size, xs.reshape((xs.size,)))
        conv.SetVecData(xv.size, xv.reshape((xv.size,)))
        conv.Fill()

    conv.Write()

def create_root_tree(f, start, end):
    file = ROOT.TFile('tree.root', 'RECREATE')
    t = ROOT.TTree('tree', 'tree')
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

#f = h5py.File("../data/train.h5", "r")
f = h5py.File("../data/test.h5", "r")

l = f[list(f.keys())[0]].shape[0]
l = 100_000
step_size = 100_000

#start = 0
#while start < l:
#    print(f"{start = }")
#    create_root_tree(f, start, start+step_size)
#
#    start += step_size

create_root_tree_fast(f,0, 1_000_000,100_000)
