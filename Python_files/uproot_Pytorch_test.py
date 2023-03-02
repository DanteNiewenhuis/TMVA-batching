import uproot
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time

from batch_generator import GetPyTorchDataLoader


def h52numpy():
    file = uproot.open("../data/Higgs_data_full.root")
    tree = file["test_tree"]
    branches = tree.arrays()
    target = "Type"

    res = []

    y = []
    for k in tree.keys():
        if k == target:
            y = branches[k].to_numpy()
            continue
        res.append(branches[k].to_numpy())

    data = np.array(res)
    X = data.transpose()

    return X, y

file_name = "../data/Higgs_data_full.root"
tree_name = "test_tree"

batch_size = 1024

X, y = h52numpy()



class BatchLoader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index: int):
        return self.X[index, :], self.y[index]

ds = BatchLoader(X, y)

loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

###################################################################################################
## AI example
###################################################################################################

def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


# Initialize pytorch 
model = torch.nn.Sequential(
    torch.nn.Linear(28, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 1),
    torch.nn.Sigmoid()
)
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



timings = []
start = time.time()

i = 0
for x_train, y_train in loader:
    
    # Make prediction and calculate loss
    pred = model(x_train).view(-1)
    loss = loss_fn(pred, y_train)

    # improve model
    model.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = calc_accuracy(y_train, pred)

    i += 1

    end = time.time()
    timings.append(end - start)
    start = time.time()

with open(f"../results/performance/PyTorch_uproot_train.csv", "w") as wf:
    for timing in timings:
        wf.write(f"{timing}\n")
