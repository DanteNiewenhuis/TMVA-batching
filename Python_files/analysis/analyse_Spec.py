# %%
import numpy as np
import matplotlib.pyplot as plt

main_folder = "../../"

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
    with open(f"{main_folder}results/DataFrame_DatasetSpec/{data_type}_{data_set}_{chunk_size}.txt", "r") as f:
        for line in f.readlines():
            data.append(float(line))

    data = np.array(data)

    plt.plot([x*chunk_size for x in range(len(data))], data)
    plt.xlabel("Starting Event")
    plt.ylabel("Loading Time(s)") 
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.savefig(f"{main_folder}results/DataFrame_DatasetSpec/Images/{data_type}_{data_set}_{chunk_size}.png")
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
with open(f"{main_folder}results/DataFrame_DatasetSpec/{data_type}_{data_set}_{chunk_size}.txt", "r") as f:
    for line in f.readlines():
        data_spec.append(float(line))

chunk_size = 10_000
data_type = "DataFrame"
data_set = "Higgs"

data_frame = []
with open(f"{main_folder}results/DataFrame_DatasetSpec/{data_type}_{data_set}_{chunk_size}.txt", "r") as f:
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


