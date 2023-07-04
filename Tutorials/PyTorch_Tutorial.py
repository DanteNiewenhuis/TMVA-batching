import torch
import time
import numpy as np
import ROOT

ROOT.EnableThreadSafety()

main_folder = "/home/dante-niewenhuis/Documents/TMVA-batching"

tree_name = "test_tree"
file_name = f"{main_folder}/data/Higgs_data_full.root"

chunk_size = 1_000_000
batch_size = 1024

gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
    tree_name,
    file_name,
    batch_size,
    chunk_size,
    target="Type",
    validation_split=0.1,
)


def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


columns = gen_train.columns

print(columns)
num_columns = len(columns)


# Initialize pytorch
model = torch.nn.Sequential(
    torch.nn.Linear(num_columns - 3, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 1),
    torch.nn.Sigmoid(),
)
loss_fn = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


for i, (x_train, y_train) in enumerate(gen_train):
    # Make prediction and calculate loss
    pred = model(x_train).view(-1)
    loss = loss_fn(pred, y_train)

    # improve model
    model.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = calc_accuracy(y_train, pred)

    print(f"{accuracy = }")
