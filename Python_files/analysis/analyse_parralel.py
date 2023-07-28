# %%

import numpy as np

import matplotlib.pyplot as plt

main_folder = "../../"
# %%

delay = 100000

parallel = []
with open(f"{main_folder}results/Parallel/threaded_{delay}.csv", "r") as rf:
    for line in rf.readlines():
        parallel.append(float(line))

parallel = np.array(parallel)

single = []
with open(f"{main_folder}results/Parallel/not_threaded_{delay}.csv", "r") as rf:
    for line in rf.readlines():
        single.append(float(line))

single = np.array(single)

normal = []
with open(f"{main_folder}results/Parallel/normal_{delay}.csv", "r") as rf:
    for line in rf.readlines():
        normal.append(float(line))

normal = np.array(normal)


# %%

plt.plot(parallel[:5000], label="Parallel")
plt.plot(single[:5000], label="Single")
plt.plot(normal[:5000], label="Normal")
plt.legend()
plt.show()

# %%

diff_par = parallel[1:] - parallel[:-1]
diff_single = single[1:] - single[:-1]
diff_normal = normal[1:] - normal[:-1]

# %%
plt.plot(diff_normal)

# %%


delay = 0

parallel = []
with open(
    f"{main_folder}results/Parallel/python_parallel_{delay*1_000_000:.0f}.csv", "r"
) as rf:
    for line in rf.readlines():
        parallel.append(float(line))

parallel = np.array(parallel)
single = []
with open(
    f"{main_folder}results/Parallel/python_single_{delay*1_000_000:.0f}.csv", "r"
) as rf:
    for line in rf.readlines():
        single.append(float(line))

single = np.array(single)

# plt.plot(parallel, label="parallel")
plt.plot(single, label="single thread")
plt.legend(fontsize=15, loc="lower right")
plt.title("Loading time of a batch using a single thread", fontsize=15)
plt.tight_layout()
plt.savefig(
    f"{main_folder}results/Images/Parallel/Parallel_vs_Single{delay*1_000_000:.0f}.png"
)
plt.show()

# %%
parallel_norm = parallel - (np.arange(len(parallel)) * delay)
single_norm = single - (np.arange(len(single)) * delay)

plt.plot(parallel_norm, label="parallel")
plt.plot(single_norm, label="single")
plt.show()
# %%

delays = [0.0001, 0.001, 0.01]

data = []

for delay in delays:
    parallel = []

    with open(
        f"{main_folder}results/Parallel/python_parallel_{delay*1_000_000:.0f}.csv", "r"
    ) as rf:
        for line in rf.readlines():
            parallel.append(float(line))

    parallel = np.array(parallel)

    data.append(parallel)

print(len(data))


# %%
norms = []

for delay, line in zip(delays, data):
    norm = line - np.arange(len(line)) * delay
    norms.append(norm)
    plt.plot(norm, label=f"processing time: {delay}")

# plt.plot(data[0], label=delays[0])

plt.legend(fontsize=15, loc="lower right")

plt.title("Batch loading time with different processing times", fontsize=15)
plt.tight_layout()
plt.savefig(f"{main_folder}results/Images/Parallel_comparison.png")

plt.show()

# %%

parallel_norm = parallel - np.arange(len(parallel)) * delay
single_norm = single - np.arange(len(single)) * delay

plt.plot(parallel_norm, label="parallel")
plt.plot(single_norm, label="single")
plt.show()
# %%

plt.hist(norms[2])
plt.hist(norms[3])

plt.show()
# %%
