# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

with open("../results/data.txt", "r") as f:
    s = np.array([float(x) for x in f.read().split(",")])

# %%

plt.plot(s)
plt.show()

# %%

first = s[:-1]
second = s[1:]
diff = second - first

epoch_size = 2250
# %%

plt.plot(diff)
plt.show()

# %%

loading = diff[np.where(diff > 0.01)]

plt.bar(range(len(loading)), loading)
plt.show()

# %%

batching = diff[np.where(diff <= 0.01)]
# %%
batching
# %%
sum(diff[1:50])
# %%
