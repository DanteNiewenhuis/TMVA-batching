# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
axis_fontsize = 18

df = pd.read_csv("results/memory.csv")
chunk_size = 1_000_000
main_folder = "/home/dante/Documents/TMVA-batching"


def get_size(num):
    file_name = f"{main_folder}/data/file_sizes_bench/{num}.root"

    return os.stat(file_name).st_size


df["size"] = df.apply(lambda row: get_size(row["num"]), axis=1)
# %%

line_width = 2.5

df = df[df["num"] < 13]

df_ROOT = df[df["method"] == "ROOT"]
df_uproot = df[df["method"] == "uproot"]

plt.plot(df_uproot["size"], df_uproot["maxresident"],
         label="uproot-tensorflow", linewidth=line_width)

labels = ["RBG 10K", "RBG 100K", "RBG 1M", "RBG 2M", "RBG 10M", "RBG 20M"]

i = 0
for cs, rows in df_ROOT.groupby("chunksize"):

    label = labels[i]
    i += 1
    plt.plot(rows["size"], rows["maxresident"],
             label=label, linewidth=line_width)

plt.xlabel("file size (b)", fontsize=axis_fontsize)
plt.ylabel("memory usage (b)", fontsize=axis_fontsize)

plt.tick_params(axis='both', which='major', labelsize=13)
# plt.tick_params(axis='both', which='minor', labelsize=10)

left_lim, right_lim = plt.xlim()
bot_lim, top_lim = plt.ylim()

plt.xlim(0, right_lim)
plt.ylim(0, top_lim)

plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("images/scaling_memory.svg", format="svg")

# %%

load_times_ROOT = []
load_time_uproot = []

chunk_sizes = [10_000, 100_000, 1_000_000, 2_000_000]

# add ROOT

for cs in chunk_sizes:
    res = []
    for num in range(1, 14):
        with open(f"results/Tensorflow_ROOT_{num}_{cs}_train.csv", "r") as rf:
            res.append(float(rf.readlines()[1].strip()))

    load_times_ROOT.append(res)

for num in range(1, 14):
    # add uproot
    with open(f"results/Tensorflow_uproot_{num}_train.csv", "r") as rf:
        load_time_uproot.append(float(rf.readlines()[0].strip()))

load_10mil = []
for num in range(5, 14):
    with open(f"results/Tensorflow_ROOT_{num}_10000000_train.csv", "r") as rf:
        load_10mil.append(float(rf.readlines()[1].strip()))

load_20mil = []
for num in range(10, 14):
    with open(f"results/Tensorflow_ROOT_{num}_20000000_train.csv", "r") as rf:
        load_20mil.append(float(rf.readlines()[1].strip()))

file_sizes = [get_size(i) for i in range(1, 14)]

plt.plot(file_sizes, load_time_uproot, label="uproot-tensorflow")

for cs, load_times in zip(chunk_sizes, load_times_ROOT):
    plt.plot(file_sizes, load_times, label=f"RGB {cs:.0f}")


# plt.plot([get_size(i) for i in range(5,14)], load_10mil, label=f"ROOT {10_000_000:.0f}")
# plt.plot([get_size(i) for i in range(10,14)], load_20mil, label=f"ROOT {20_000_000:.0f}")


plt.tick_params(axis='both', which='major', labelsize=13)

plt.ylabel("loading time (s)", fontsize=axis_fontsize)
plt.xlabel("file size (byte)", fontsize=axis_fontsize)

plt.legend()
plt.tight_layout()
plt.savefig("images/scaling_loading_time.svg", format="svg")

# %%

#########################################################################################
# histograms

df_uproot = pd.read_csv(
    f"results/Tensorflow_uproot_5_train.csv", header=None, names=["time"])
df_ROOT = pd.read_csv(
    f"results/Tensorflow_ROOT_5_1000000_train.csv", header=None, names=["time"])

plt.hist(df_uproot[(df_uproot["time"] < 0.01) & (
    df_uproot["time"] > 0)]["time"]*1000, bins=100, label="TensorFlow", alpha=.75)
plt.hist(df_ROOT[(df_ROOT["time"] < 0.01) & (df_ROOT["time"] > 0)]
         ["time"]*1000, bins=100, label="RBatchGenerator", alpha=.75)

plt.xlabel("batch processing time (ms)", fontsize=axis_fontsize)
plt.ylabel("count", fontsize=axis_fontsize)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("images/loading_hist.svg", format="svg")

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
plt.savefig("images/ROOT_vs_uproot.svg", format="svg")


# %%

###########################################################################
# Single vs Parallel
###########################################################################

df_single = pd.read_csv(
    f"results/python_single_100000.csv", header=None, names=["time"])
df_parallel = pd.read_csv(
    f"results/python_parallel_100000.csv", header=None, names=["time"])


df_single["time_norm"] = df_single["time"] - (0.1 * df_single.index)
df_parallel["time_norm"] = df_parallel["time"] - (0.1 * df_parallel.index)

# %%

plt.plot(df_single["time_norm"], label="single", linewidth=2.5)
plt.plot(df_parallel["time_norm"], label="parallel", linewidth=2.5)

plt.xlabel("batch number", fontsize=axis_fontsize)
plt.ylabel("time (s)", fontsize=axis_fontsize)

plt.legend(fontsize=18)

plt.tight_layout()
plt.savefig(f"images/single_vs_parallel.svg", format="svg")
# %%

df_single["time_norm"]

# %%


axis_fontsize = 18

df = pd.read_csv("results/memory.csv")
chunk_size = 1_000_000
main_folder = "/home/dante/Documents/TMVA-batching"


# %%

line_width = 4
df_uproot = df[df["method"] == "uproot"]
df_ROOT = df[df["chunksize"] == 10000]

df_uproot = df_uproot[df_uproot["num"] < 13]
df_ROOT = df_ROOT[df_ROOT["num"] < 13]


plt.plot(df_uproot["num"] * .125 * 10**9, df_uproot["maxresident"],
         label="uproot", linewidth=line_width)

plt.plot(df_ROOT["num"] * .125 * 10**9, df_ROOT["maxresident"],
         label="RBatchGenerator", linewidth=line_width)

plt.xlabel("file size (b)", fontsize=axis_fontsize)
plt.ylabel("memory usage (b)", fontsize=axis_fontsize)

plt.tick_params(axis='both', which='major', labelsize=13)
# plt.tick_params(axis='both', which='minor', labelsize=10)

left_lim, right_lim = plt.xlim()
bot_lim, top_lim = plt.ylim()

plt.xlim(0, right_lim)
plt.ylim(0, top_lim)

plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("images/scaling_memory_1.svg", format="svg")

# def get_size(num):
#     file_name = f"{main_folder}/data/file_sizes_bench/{num}.root"

#     return os.stat(file_name).st_size

# df["size"] = df.apply(lambda row: get_size(row["num"]), axis=1)
# %%

line_width = 2.5

df = df[df["num"] < 13]

df_ROOT = df[df["method"] == "ROOT"]
df_uproot = df[df["method"] == "uproot"]

plt.plot(df_uproot["size"], df_uproot["maxresident"],
         label="uproot-tensorflow", linewidth=line_width)

labels = ["RBG 10K", "RBG 100K", "RBG 1M", "RBG 2M", "RBG 10M", "RBG 20M"]

i = 0
for cs, rows in df_ROOT.groupby("chunksize"):

    label = labels[i]
    i += 1
    plt.plot(rows["size"], rows["maxresident"],
             label=label, linewidth=line_width)

plt.xlabel("file size (b)", fontsize=axis_fontsize)
plt.ylabel("memory usage (b)", fontsize=axis_fontsize)

plt.tick_params(axis='both', which='major', labelsize=13)
# plt.tick_params(axis='both', which='minor', labelsize=10)

left_lim, right_lim = plt.xlim()
bot_lim, top_lim = plt.ylim()

plt.xlim(0, right_lim)
plt.ylim(0, top_lim)

plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("images/scaling_memory.svg", format="svg")
