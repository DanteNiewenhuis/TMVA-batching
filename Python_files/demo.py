import ROOT
from batch_generator_spec import Generator
import torch

main_folder = "../"

columns = ["m_jj", "m_jjj", "m_jlv"] 

tree_name = "sig_tree"
file_name = f"{main_folder}data/h5train_combined.root"

num_columns = len(columns)
batch_rows = 1054
chunk_rows = 500_000

generator = Generator(file_name, tree_name, columns, chunk_rows, batch_rows, use_whole_file=True)

for i, batch in enumerate(generator):
    print(f"batch {i}, {batch}")


###################################################################################################
## AI example
###################################################################################################

def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


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
    x_train, y_train = batch[:,:num_columns-1], batch[:, -1]
    
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
