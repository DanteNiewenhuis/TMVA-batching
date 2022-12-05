import ROOT
from batch_generator_parralel import Generator

main_folder = "../"

columns = ["m_jj", "m_jjj", "m_jlv"] 
# x_rdf = ROOT.RDataFrame("sig_tree", f"{main_folder}data/Higgs_data_full.root", columns)
x_rdf = ROOT.RDataFrame("sig_tree", f"{main_folder}data/r0-20.root", columns)

x_filter = x_rdf.Filter("m_jj < 1")

num_columns = len(columns)
batch_rows = 2
chunk_rows = 5

generator = Generator(x_filter, columns, chunk_rows, batch_rows, use_whole_file=True)

for i, batch in enumerate(generator):
    print(f"batch {i}, {batch}")

raise NotImplementedError

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
