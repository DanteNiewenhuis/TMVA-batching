# %% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

# %%

df = pd.read_csv("results/memory.csv")
chunk_size = 1_000_000
main_folder = "/home/danteniewenhuis/Documents/CERN/TMVA-batching"


def get_size(num):
    file_name = f"{main_folder}/data/file_sizes_bench/{num}.root"

    return os.stat(file_name).st_size

df["size"] = df.apply(lambda row: get_size(row["num"]), axis=1)
# %%

df = df[df["num"] < 13]

df_ROOT = df[df["method"] == "ROOT"]
df_uproot = df[df["method"] == "uproot"]

plt.plot(df_uproot["size"], df_uproot["maxresident"], label="uproot")

for cs, rows in df_ROOT.groupby("chunksize"):
    label = f'ROOT {cs:.0f}'
    # while len(label) < 12:
    #     label = label[:4] + " " + label[4:]
    plt.plot(rows["size"], rows["maxresident"], label=label)

plt.xlabel("file size (b)")
plt.ylabel("memory usage (b)")

left_lim, right_lim = plt.xlim()
bot_lim, top_lim = plt.ylim()

plt.xlim(0,right_lim)
plt.ylim(0,top_lim)

plt.legend()
plt.tight_layout()

plt.savefig("images/scaling_memory.png")

# %%

load_times_ROOT = []
load_time_uproot = []

chunk_sizes = [10_000, 100_000, 1_000_000, 2_000_000]


    # add ROOT
    
for cs in chunk_sizes:
    res = []
    for num in range(1,14):
        with open(f"results/Tensorflow_ROOT_{num}_{cs}_train.csv", "r") as rf:
            res.append(float(rf.readlines()[1].strip()))

    load_times_ROOT.append(res)
    
for num in range(1,14):
    # add uproot
    with open(f"results/Tensorflow_uproot_{num}_train.csv", "r") as rf:
        load_time_uproot.append(float(rf.readlines()[0].strip()))

load_10mil = []
for num in range(5,14):
    with open(f"results/Tensorflow_ROOT_{num}_10000000_train.csv", "r") as rf:
        load_10mil.append(float(rf.readlines()[1].strip()))

load_20mil = []
for num in range(10,14):
    with open(f"results/Tensorflow_ROOT_{num}_20000000_train.csv", "r") as rf:
        load_20mil.append(float(rf.readlines()[1].strip()))

file_sizes = [get_size(i) for i in range(1,14)]

plt.plot(file_sizes, load_time_uproot, label="uproot")

for cs, load_times in zip(chunk_sizes, load_times_ROOT):
    plt.plot(file_sizes, load_times, label=f"ROOT {cs:.0f}")


# plt.plot([get_size(i) for i in range(5,14)], load_10mil, label=f"ROOT {10_000_000:.0f}")
# plt.plot([get_size(i) for i in range(10,14)], load_20mil, label=f"ROOT {20_000_000:.0f}")


plt.ylabel("loading time (s)")
plt.xlabel("file size")

plt.legend()
plt.tight_layout()
plt.savefig("images/scaling_loading_time.png")

# %%

#########################################################################################
# histograms

df_uproot = pd.read_csv(f"results/Tensorflow_uproot_5_train.csv", header=None, names=["time"])
df_ROOT = pd.read_csv(f"results/Tensorflow_ROOT_5_1000000_train.csv", header=None, names=["time"])

plt.hist(df_uproot[(df_uproot["time"] < 0.01) & (df_uproot["time"] > 0)]["time"], bins=100, label="uproot", alpha=.75)
plt.hist(df_ROOT[(df_ROOT["time"] < 0.01) & (df_ROOT["time"] > 0)]["time"], bins=100, label="ROOT", alpha=.75)

plt.xlabel("batch loading time (s)")

plt.legend()

plt.tight_layout()
plt.savefig("images/loading_hist.png")

# %%

#########################################################################################
# processing time

cum_uproot = [0]

for i, row in df_uproot.iterrows():
    cum_uproot.append(cum_uproot[-1] + row.item())

cum_uproot = cum_uproot[1:]

cum_ROOT = [0]

for i, row in df_uproot.iterrows():
    cum_ROOT.append(cum_ROOT[-1] + row.item())

cum_ROOT = cum_ROOT[1:]

# %%

plt.plot(range(len(cum_ROOT)), cum_ROOT, label="ROOT", alpha=0.75)
plt.plot(range(len(cum_uproot)), cum_uproot, label="uproot", alpha=0.75)

plt.legend()

plt.tight_layout()
plt.savefig("images/ROOT_vs_uproot.png")


# %%

###########################################################################
# Single vs Parallel
###########################################################################

df_single = pd.read_csv(f"results/python_single_100000.csv", header=None, names=["time"])
df_parallel = pd.read_csv(f"results/python_parallel_100000.csv", header=None, names=["time"])


df_single["time_norm"] = df_single["time"] - (0.1 * df_single.index)
df_parallel["time_norm"] = df_parallel["time"] - (0.1 * df_parallel.index)

# %%

plt.plot(df_single["time_norm"], label="single")
plt.plot(df_parallel["time_norm"], label="parallel")

plt.xlabel("batch number")
plt.ylabel("time (s)")

plt.legend()

plt.tight_layout()
plt.savefig(f"images/single_vs_parallel.png")
# %%

df_single["time_norm"]