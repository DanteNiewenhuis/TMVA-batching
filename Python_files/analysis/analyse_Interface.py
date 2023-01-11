# %%
import numpy as np
import matplotlib.pyplot as plt

main_folder = "../../"

# %%

data_Cpp = []
chunk_size = 1_000_000
alg = "_Spec"

with open(f"{main_folder}results/Cpp_Python/{chunk_size}_Cpp{alg}.txt", "r") as f:
    for line in f.readlines():
        data_Cpp.append(float(line))

data_Cpp = np.array(data_Cpp)

data_Python_Cpp = []

with open(f"{main_folder}results/Cpp_Python/{chunk_size}_Python_Cpp{alg}.txt", "r") as f:
    for line in f.readlines():
        data_Python_Cpp.append(float(line))

data_Python_Cpp = np.array(data_Python_Cpp[:len(data_Cpp)])

data_Python = []

with open(f"{main_folder}results/Cpp_Python/{chunk_size}_Python{alg}.txt", "r") as f:
    for line in f.readlines():
        data_Python.append(float(line))

data_Python = np.array(data_Python[:len(data_Cpp)])

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)

plt.rc('axes', labelsize=20)
line_width=2

plt.plot(data_Cpp, linewidth=line_width, label = "C++")
plt.plot(data_Python, linewidth=line_width, label = "Python")
plt.plot(data_Python_Cpp, linewidth=line_width, label = "Hybrid")
plt.legend(prop={'size': 15})

plt.tight_layout()
plt.savefig(f"{main_folder}results/Images/Interface/{chunk_size}.png")

plt.show()
# %%


diff_Cpp = data_Cpp[1:] - data_Cpp[:-1]
diff_Python = data_Python[1:] - data_Python[:-1]
diff_Python_Cpp = data_Python_Cpp[1:] - data_Python_Cpp[:-1]

# %%
num_batches = 100
chunks = range(0, len(diff_Python), num_batches)
batches = [x for x in range(len(diff_Python)) if x % num_batches != 0]
 
# %%

plt.hist(diff_Cpp[chunks], alpha=0.8, label = "C++")
plt.hist(diff_Python[chunks], alpha=0.8, label = "Python")
plt.hist(diff_Python_Cpp[chunks], alpha=0.8, label = "Hybrid")
plt.legend()
plt.show()

# %%


print(diff_Cpp[-1])
