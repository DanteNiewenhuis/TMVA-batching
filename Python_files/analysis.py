# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)

plt.rc('axes', labelsize=20)

for chunk_size in [10_000, 20_000, 50_000, 100_000]:
# chunk_size = 200_000
    data_type = "DataFrame"
    # data_type = "DatasetSpec"
    data_set = "Higgs"

    data = []
    with open(f"../results/DataFrame_DatasetSpec/{data_type}_{data_set}_{chunk_size}.txt", "r") as f:
        for line in f.readlines():
            data.append(float(line))

    data = np.array(data)

    plt.plot([x*chunk_size for x in range(len(data))], data)
    plt.xlabel("Starting Event")
    plt.ylabel("Loading Time(s)") 
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.savefig(f"../results/DataFrame_DatasetSpec/Images/{data_type}_{data_set}_{chunk_size}.png")
    # plt.title(f"{chunk_size = }")

    plt.show()

# %%

data[425]
# %%
data[25]

# %%

avg_data = []

for i in range(int(len(data)/5)):
    avg_data.append(np.mean(data[i:i+5]))


# %%

plt.plot([x*50_000 for x in range(len(avg_data))], avg_data)
plt.show()
# %%


chunk_size = 10_000
data_type = "DatasetSpec"
data_set = "Higgs"

data_spec = []
with open(f"../results/DataFrame_DatasetSpec/{data_type}_{data_set}_{chunk_size}.txt", "r") as f:
    for line in f.readlines():
        data_spec.append(float(line))

chunk_size = 10_000
data_type = "DataFrame"
data_set = "Higgs"

data_frame = []
with open(f"../results/DataFrame_DatasetSpec/{data_type}_{data_set}_{chunk_size}.txt", "r") as f:
    for line in f.readlines():
        data_frame.append(float(line))
# %%

plt.plot([x*chunk_size for x in range(len(data))], data_spec, label="DatasetSpec")
plt.plot([x*chunk_size for x in range(len(data))], data_frame, label="DataFrame")
plt.xlabel("Starting Event")
plt.ylabel("Loading Time (s)")
plt.ylim(ymin=0)
plt.legend()
plt.show()
# %%

data_Cpp = []
chunk_size = 200_000
alg = "_Spec"

with open(f"../results/Cpp_Python/{chunk_size}_Cpp{alg}.txt", "r") as f:
    for line in f.readlines():
        data_Cpp.append(float(line))

data_Cpp = np.array(data_Cpp)

data_Python_Cpp = []

with open(f"../results/Cpp_Python/{chunk_size}_Python_Cpp{alg}.txt", "r") as f:
    for line in f.readlines():
        data_Python_Cpp.append(float(line))

data_Python_Cpp = np.array(data_Python_Cpp)

data_Python = []

with open(f"../results/Cpp_Python/{chunk_size}_Python{alg}.txt", "r") as f:
    for line in f.readlines():
        data_Python.append(float(line))

data_Python = np.array(data_Python)

# %%

plt.plot(data_Cpp, label = "Cpp")
plt.plot(data_Python, label = "Python")
plt.plot(data_Python_Cpp, label = "Python Cpp")
plt.legend()
plt.show()

# %%

data_Python_Frame = []

with open("../results/Cpp_Python/Python.txt", "r") as f:
    for line in f.readlines():
        data_Python_Frame.append(float(line))

data_Python_Frame = np.array(data_Python_Frame)

data_Python = []

with open("../results/Cpp_Python/Python_Spec.txt", "r") as f:
    for line in f.readlines():
        data_Python.append(float(line))

data_Python = np.array(data_Python)


# %%

plt.plot(data_Python, label = "Spec")
plt.plot(data_Python_Frame, label = "Frame")
plt.legend()
plt.show()

# %%

data_Cpp_Frame = []
with open("../results/chunk_test/Cpp_Frame.txt", "r") as f:
    for line in f.readlines():
        data_Cpp_Frame.append(float(line))

data_Cpp_Frame = np.array(data_Cpp_Frame)

data_Cpp_Spec = []
with open("../results/chunk_test/Cpp_Spec.txt", "r") as f:
    for line in f.readlines():
        data_Cpp_Spec.append(float(line))

data_Cpp_Spec = np.array(data_Cpp_Spec)


# %%

plt.plot(data_Cpp_Frame, label = "Frame")
plt.plot(data_Cpp_Spec, label = "Spec")
plt.legend()
plt.show()
# %%

data_Cpp_Frame = []
with open("../results/DataFrame_DatasetSpec/DataFrame_h5_500000.txt", "r") as f:
    for line in f.readlines():
        data_Cpp_Frame.append(float(line))

data_Cpp_Frame = np.array(data_Cpp_Frame)

data_Cpp_Spec = []
with open("../results/DataFrame_DatasetSpec/DatasetSpec_h5_500000.txt", "r") as f:
    for line in f.readlines():
        data_Cpp_Spec.append(float(line))

data_Cpp_Spec = np.array(data_Cpp_Spec)


# %%

plt.plot(data_Cpp_Frame, label = "Frame")
plt.plot(data_Cpp_Spec, label = "Spec")
plt.legend()
plt.show()
# %%
