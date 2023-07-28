# %%

import numpy as np

import matplotlib.pyplot as plt

# %%

with open("results/python_single_100000.csv", "r") as rf:
    timings = np.array([float(x.strip()) for x in rf.readlines()])

loading_overhead = timings[1:] - timings[:-1]

loading_overhead = loading_overhead[1:]
sorted_overhead = sorted(loading_overhead)

# %%

print(loading_overhead.mean())
print(loading_overhead.max())


loading_overhead_high = loading_overhead[loading_overhead > 0.15]
loading_overhead_low = loading_overhead[loading_overhead <= 0.15]


print(loading_overhead_high)
print(loading_overhead_low)


# %%


# %%


print(loading_overhead.mean())
print(loading_overhead.max())


loading_overhead_high = loading_overhead[loading_overhead > 0.15]
loading_overhead_low = loading_overhead[loading_overhead <= 0.15]


print(loading_overhead_high)
print(loading_overhead_low)

# %%


def print_overhead(loading_overhead: np.ndarray):
    chunk_idx = [list(range(194, len(loading_overhead), 195))]
    batch_idx = [x for x in range(len(loading_overhead)) if x not in chunk_idx]
    print(f"{loading_overhead.mean() = }")
    print(f"{loading_overhead.max() = }")

    loading_overhead_high: np.ndarray = loading_overhead[chunk_idx][0]
    loading_overhead_low: np.ndarray = loading_overhead[batch_idx][0]

    if len(loading_overhead_high) > 0:
        print(f"{loading_overhead_high.mean() = }")

    print(f"{loading_overhead_low.mean() = }")

    return loading_overhead_high.mean()


# %%

res = []

with open("results/Parallel/python_single_100000.csv", "r") as rf:
    timings = np.array([float(x.strip()) for x in rf.readlines()])

loading_overhead = timings[1:] - timings[:-1]
loading_overhead = loading_overhead[1:]

res.append(print_overhead(loading_overhead))


for delay in [0, 100, 1000, 10_000, 100_000]:
    print(f"\n{delay = }")
    with open(f"results/Parallel/python_parallel_{delay}.csv", "r") as rf:
        timings = np.array([float(x.strip()) for x in rf.readlines()])

    loading_overhead = timings[1:] - timings[:-1]
    loading_overhead = loading_overhead[1:]

    res.append(print_overhead(loading_overhead))


# %%

plt.bar(["single", str(0), str(100), str(1000), str(10_000), str(100_000)], res)

# %%


with open("results/Tensorflow_ROOT_train.csv", "r") as rf:
    res_ROOT = [float(x.strip()) for x in rf.readlines()]

with open("results/Tensorflow_uproot_train.csv", "r") as rf:
    res_uproot = [float(x.strip()) for x in rf.readlines()]


# %%

res_ROOT = res_ROOT[2:]
res_uproot = res_uproot[2:-2]

# %%

print(len(res_ROOT))
print(len(res_uproot))

with open("results/combined.csv", "w") as wf:
    wf.write("ROOT,uproot\n")

    for x, y in zip(res_ROOT, res_uproot):
        wf.write(f"{x},{y}\n")
