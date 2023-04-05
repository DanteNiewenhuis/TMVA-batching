# %%

import numpy as np
import matplotlib.pyplot as plt

# %%


with open("results/Tensorflow_ROOT_train.csv", "r") as rf:

    data_ROOT = np.array([float(x.strip()) for x in rf.readlines()])


with open("results/Tensorflow_uproot_train.csv", "r") as rf:

    data_uproot = np.array([float(x.strip()) for x in rf.readlines()])

# %%

data_cum_ROOT = [data_ROOT[0]]

for i in range(1, len(data_ROOT)):
    data_cum_ROOT.append(data_cum_ROOT[-1]+data_ROOT[i])

data_cum_uproot = [data_uproot[0]]

for i in range(1, len(data_uproot)):
    data_cum_uproot.append(data_cum_uproot[-1]+data_uproot[i])
# %%

plt.plot(data_cum_ROOT, label="ROOT")
plt.plot(data_cum_uproot, label="uproot")

plt.legend()

plt.show()

# %%

plt.hist(data_ROOT[data_ROOT < 0.01], alpha=.5, bins=20, label="ROOT")
plt.hist(data_uproot[data_uproot < 0.01], alpha=.5, bins=20, label="uproot")

plt.legend()
plt.show()

# %%
