# %%

import numpy as np

import matplotlib.pyplot as plt

main_folder = "../../"
# %%

delay = 0.1

parralel = []
with open(f"{main_folder}/results/Parralel/parralel_{delay}.csv", "r") as rf:
    for line in rf.readlines():
        parralel.append(float(line))

parralel = np.array(parralel)

# normal = []
# with open(f"{main_folder}/results/Parralel/normal_{delay}.csv", "r") as rf:
#     for line in rf.readlines():
#         normal.append(float(line))

# normal = np.array(normal)

# %%

plt.plot(parralel)
# plt.plot(normal)

plt.show()
# %%

diff_parralel = parralel[1:] - parralel[:-1]
# diff_normal = normal[1:] - normal[:-1]


loading_parralel = np.array([diff_parralel[i] for i in range(len(diff_parralel)) if i%195 == 0])
# loading_normal = np.array([diff_normal[i] for i in range(len(diff_normal)) if i%195 == 0])

batching_parralel = np.array([diff_parralel[i] for i in range(len(diff_parralel)) if i%195 != 0])
# batching_normal = np.array([diff_normal[i] for i in range(len(diff_normal)) if i%195 != 0])


# %%

print(loading_parralel)

# %%
