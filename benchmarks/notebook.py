# %%

import h5py
from tensorflow.data import Dataset

import numpy as np

# %%

file = h5py.File("../data/test.h5")
list(file.keys())
# %%

columns = ['fjet_C2',
 'fjet_D2',
 'fjet_ECF1',
 'fjet_ECF2',
 'fjet_ECF3',
 'fjet_L2',
 'fjet_L3',
 'fjet_Qw',
 'fjet_Split12',
 'fjet_Split23',
 'fjet_Tau1_wta',
 'fjet_Tau2_wta',
 'fjet_Tau3_wta',
 'fjet_Tau4_wta',
 'fjet_ThrustMaj',
 'fjet_clus_E',
 'fjet_eta',
 'fjet_m',
 'fjet_phi',
 'fjet_pt',
 'labels',
 'weights']

# %%

dset = file['fjet_clus_E']


# %%

features = []

for c in columns:
    features.append(file[c])

stacked = np.stack(features, axis=-1)

# %%
data_set = Dataset.from_tensor_slices(stacked).batch(1024)

# %%

for b in data_set:
    print(b)

# %%

print(type(stacked)) 
# %%

dset = file["fjet_C2"]

# %%

data_set = Dataset.from_tensor_slices(dset).batch(10)

# %%


