# %%
import numpy as np
import matplotlib.pyplot as plt

main_folder = "../../"

# %% 

# External batchgenerator tester

delays = [0, 1, 10, 100, 1000]
data = []

for delay in delays:
    graph = []
    with open(f"{main_folder}results/threaded_batching/batching{delay}.txt", "r") as f:
        for line in f.readlines():
            graph.append(float(line))

    graph = np.array(graph)

    graph_min = graph - (np.arange(len(graph)) * (delay / 1_000_000))
    
    data.append(graph_min)
# diff = line[1:] - line[:-1]

# %%



# %%

for graph, delay in zip(data, delays):
    print("plot")
    plt.plot(graph, label=f"{delay}")

plt.legend()
plt.show()


# %%

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
