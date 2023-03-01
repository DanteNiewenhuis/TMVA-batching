import ROOT
from batch_generator import GetGenerators, GetPyTorchDataLoader
import torch

import time

main_folder = "../"


tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

# columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
#                                 "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
#                                 "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
#                                 "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
#                                 "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights", "labels"]
# filters = ["fjet_D2 < 5"]
filters = []

batch_rows = 1024
chunk_rows = 1_000_000

train_generator, test_generator = GetPyTorchDataLoader(file_name, tree_name, chunk_rows,
                                                     batch_rows, validation_split=0, target="Type")

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

start = time.time()
batch_timings = []
train_timings = []
i = 0
for x_train, y_train in train_generator():
    end = time.time()
    batch_timings.append(end-start)
    start = time.time()

    # Make prediction and calculate loss
    pred = model(x_train).view(-1)
    loss = loss_fn(pred, y_train)

    # improve model
    model.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = calc_accuracy(y_train, pred)

    # if i % 100 == 0:
    #     print(f"train batch {i}: {loss.item():.4f} --- {accuracy:.4f}")

    i += 1
    
    end = time.time()
    train_timings.append(end-start)
    start = time.time()

with open(f"../results/performance/PyTorch_ROOT_batching.csv", "w") as wf:
    for timing in batch_timings:
        wf.write(f"{timing}\n")

with open(f"../results/performance/PyTorch_ROOT_training.csv", "w") as wf:
    for timing in train_timings:
        wf.write(f"{timing}\n")

# print("Evaluation!")

# start = time.time()
# i = 0
# with torch.no_grad():
#     for x_test, y_test in test_generator():
#         print(f"testing batch took {end - start :.6f}")
        
#         # Make prediction and calculate loss
#         pred = model(x_test).view(-1)

#         accuracy = calc_accuracy(y_test, pred)

#         if i % 100 == 0:
#             print(f"test batch {i}: {loss.item():.4f} --- {accuracy:.4f}")

#         end = time.time()
#         i += 1

#         start = time.time()

# print("Finished")