import ROOT
from batch_generator import Generator
import torch

main_folder = "../"


tree_name = "sig_tree"
file_name = f"{main_folder}data/h5train_combined.root"

# x_rdf = ROOT.RDataFrame(tree_name, file_name)
# columns = x_rdf.GetColumnNames()

columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"]
filters = ["fjet_D2 < 5"]
# filters = []

num_columns = len(columns)
batch_rows = 1024
chunk_rows = 200_000

generator = Generator(file_name, tree_name, columns, filters, chunk_rows, batch_rows, use_whole_file=True)

###################################################################################################
## AI example
###################################################################################################

def calc_accuracy(targets, pred):
    return torch.sum(targets.round() == pred.round()) / pred.size(0)


# Initialize pytorch 
model = torch.nn.Sequential(
    torch.nn.Linear(num_columns-1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
    torch.nn.Sigmoid()
)
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

i = 0
for batch in generator:


    # Split x and y
    x_train, y_train = torch.Tensor(batch[0]), torch.Tensor(batch[1])
    
    # Make prediction and calculate loss
    pred = model(x_train).view(-1)
    loss = loss_fn(pred, y_train)

    # improve model
    model.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = calc_accuracy(y_train, pred)

    print(f"batch {i}: {loss.item():.4f} --- {accuracy:.4f}")

    i += 1
