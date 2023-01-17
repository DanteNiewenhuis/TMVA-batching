# %%

import numpy as np
# %%

def get_target(batch, target_index):
    before = batch[:, :target_index]
    after = batch[:, target_index+1:]
    target = batch[:, target_index]

    return np.column_stack((before, after)), target


# %%

x = np.array([[1,2,3,4,5],[6,7,8,9,10]])
target_index = 4

# %%

get_target(x, target_index)
# %%
