# %%

import numpy as np

# %%

res = []

for i in range(8):
    res.append(i * 2 + np.arange(0, 10))

x = np.array(res)

# %%

all_indices = list(range(x.shape[0]))

target_indices = [1, 7, 3]
weight_index = [4]

train_indices = [c for c in all_indices if c not in target_indices + weight_index]

# %%

x[:, target_indices]

# %%

train_indices
