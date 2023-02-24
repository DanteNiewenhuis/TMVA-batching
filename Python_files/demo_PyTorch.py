import ROOT
from batch_generator import GetGenerators
import torch

import time

main_folder = "../"


tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

x_rdf = ROOT.RDataFrame(tree_name, file_name)
columns = x_rdf.GetColumnNames()

# columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
#                                 "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
#                                 "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
#                                 "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
#                                 "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights", "labels"]
# filters = ["fjet_D2 < 5"]
filters = []

num_columns = len(columns)
batch_rows = 10_000
chunk_rows = 100_000

train_generator, test_generator = GetGenerators(file_name, tree_name, chunk_rows, batch_rows, target="Type", 
                                                validation_split=0.5, use_whole_file=False, max_chunks=10)

###################################################################################################
## AI example
###################################################################################################

def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


# Initialize pytorch 
model = torch.nn.Sequential(
    torch.nn.Linear(num_columns-1, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
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

i = 0
for x, y in train_generator:
    end = time.time()
    print(f"training batch took {end - start :.6f}")
    
    # Split x and y
    x_train, y_train = torch.Tensor(x), torch.Tensor(y)
    
    # Make prediction and calculate loss
    pred = model(x_train).view(-1)
    loss = loss_fn(pred, y_train)

    # improve model
    model.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = calc_accuracy(y_train, pred)

    # print(f"train batch {i}: {loss.item():.4f} --- {accuracy:.4f}")

    i += 1

    start = time.time()

print("Evaluation!")

start = time.time()
# with torch.no_grad():
for x, y in test_generator:
    end = time.time()
    print(f"testing batch took {end - start :.6f}")
    
    # Split x and y
    x_test, y_test = torch.Tensor(x), torch.Tensor(y)
    
    # Make prediction and calculate loss
    pred = model(x_test).view(-1)

    accuracy = calc_accuracy(y_test, pred)

    # print(f"test batch {i}: {loss.item():.4f} --- {accuracy:.4f}")

    i += 1

    start = time.time()

print("Finished")