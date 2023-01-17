# %%
import numpy as np
import matplotlib.pyplot as plt

main_folder = "../../"

# %% 
# External batchgenerator tester

data_Cpp = []
chunk_size = 1_000_000
batch_size = 2000
alg = "_Spec"

data_Python = []
with open(f"../results/Cpp_Python/{chunk_size}_Python{alg}.txt", "r") as f:
    for line in f.readlines():
        data_Python.append(float(line))

data_Python = np.array(data_Python)

data_Pytorch = []

with open(f"../results/Cpp_Python/{chunk_size}_Pytorch.txt", "r") as f:
    for line in f.readlines():
        data_Pytorch.append(float(line))

data_Pytorch = np.array(data_Pytorch[:len(data_Python)])

data_TF = []

with open(f"../results/Cpp_Python/{chunk_size}_TF.txt", "r") as f:
    for line in f.readlines():
        data_TF.append(float(line))

data_TF = np.array(data_TF[:len(data_Python)])


## DIFF
diff_Python = data_Python[1:] - data_Python[:-1]
diff_Pytorch = data_Pytorch[1:] - data_Pytorch[:-1]
diff_TF = data_TF[1:] - data_TF[:-1]

t = int(chunk_size / batch_size)
loading_idx = [x for x in range(len(diff_Python)) if x % t == 0]
batching_idx = [[x for x in range(len(diff_Python)) if x % t != 0]]

loading_Python = diff_Python[loading_idx]
loading_Pytorch = diff_Pytorch[loading_idx]
loading_TF = diff_TF[loading_idx]

batching_Python = diff_Python[batching_idx]
batching_Pytorch = diff_Pytorch[batching_idx]
batching_TF = diff_TF[batching_idx]

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)

plt.rc('axes', labelsize=20)

line_width=2

x_labels = [x*batch_size for x in range(len(data_Python))]
plt.plot(x_labels, data_Python, label = "ROOT", linewidth=line_width)
plt.plot(x_labels, data_Pytorch, label = "Pytorch", linewidth=line_width)
plt.plot(x_labels, data_TF, label = "TensorFlow", linewidth=line_width)
plt.legend(prop={'size': 15})

plt.tight_layout()
plt.savefig(f"../results/Images/Root_vs_External/{chunk_size}.png")

plt.show()
