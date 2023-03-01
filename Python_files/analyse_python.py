# %%

import numpy as np
import matplotlib.pyplot as plt

# %%

batch_timing = []
with open(f"../results/performance/Tensorflow_ROOT_train.csv", "r") as rf:
    for line in rf.readlines():
        batch_timing.append(float(line))

print(batch_timing)

train_timing = []
with open(f"../results/performance/PyTorch_uproot_training.csv", "r") as rf:
    for line in rf.readlines():
        train_timing.append(float(line))

print(train_timing)


# %%

plt.plot(batch_timing[1:])

plt.show()

# %%


window = 50
average_train = []

timing = train_timing[1:]
for ind in range(len(timing) - window + 1):
    average_train.append(np.mean(timing[ind:ind+window]))

# %%

plt.plot(average_train)
plt.show()

# %%
