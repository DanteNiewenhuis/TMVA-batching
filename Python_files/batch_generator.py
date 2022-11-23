import ROOT
import torch
import numpy as np

class Generator:

    def __init__(self, x_rdf: ROOT.TMVA.Experimental.RTensor, columns: list[str], chunk_rows: int, 
                 batch_rows: int, num_chuncks: int=1, use_whole_file: bool=True):
        
        # Initialize parameters
        self.x_rdf = x_rdf
        self.columns = columns
        self.chunk_rows = chunk_rows
        self.current_chunck = 0
        self.num_chuncks = num_chuncks
        self.EoF = False

        self.num_columns = len(columns)
        self.batch_rows = batch_rows
        self.batch_size = batch_rows * self.num_columns
        self.use_whole_file = use_whole_file

        # Create C++ function
        ROOT.gInterpreter.ProcessLine("""
size_t load_data(TMVA::Experimental::RTensor<float>& x_tensor, ROOT::RDataFrame x_rdf,
                std::vector<std::string> cols, const size_t num_columns, 
                const size_t chunk_rows, const size_t start_row = 0, bool random_order=true) 
{

    
    // Fill the RTensor with the data from the RDataFrame
""" + f"DataLoader<float, std::make_index_sequence<{self.num_columns}>>" + """
        func(x_tensor, num_columns, chunk_rows, random_order);

    auto myCount = x_rdf.Range(start_row, start_row + chunk_rows).Count();

    x_rdf.Range(start_row, start_row + chunk_rows).Foreach(func, cols);

    return myCount.GetValue();
}
""")
        # Create x_tensor
        self.x_tensor = ROOT.TMVA.Experimental.RTensor("float")([self.chunk_rows, self.num_columns])    

        self.load_data()

    def load_data(self):

        if (self.EoF):
            raise StopIteration

        start = self.current_chunck * self.chunk_rows

        # Fill x_tensor
        count = ROOT.load_data(self.x_tensor, x_rdf, self.columns, self.num_columns, self.chunk_rows, start)

        #TODO: think about what to do if end of file is reached
        if (count < self.batch_rows):
            print("End of File reached")
            self.EoF = True

        # Create Generator
        self.generator = ROOT.Generator_t(self.x_tensor, count, self.chunk_rows, self.num_columns)

    def __iter__(self):
        # reset

        return self

    # Return a batch when available
    def __next__(self) -> ROOT.TMVA.Experimental.RTensor:
        
        if (self.generator.HasData()):
            batch = self.generator()
            data = batch.GetData()
            data.reshape((self.batch_size,))
            return torch.Tensor(data).view(self.batch_rows, self.num_columns)

        # Load the next chunk
        self.current_chunck += 1
        if (self.current_chunck < self.num_chuncks or self.use_whole_file):
            self.load_data()
            return self.__next__()
        
        raise StopIteration


main_folder = "../"

# Import myBatcher.C
ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}/Cpp_files/myBatcher.C"')

columns = ["m_jj", "m_jjj", "m_jlv", "Type"] 
x_rdf = ROOT.RDataFrame("sig_tree", f"{main_folder}/data/Higgs_data.root", columns)
# x_rdf = ROOT.RDataFrame("testTree", "testFile.root", columns)

num_columns = len(columns)
batch_rows = 100
chunk_rows = 100_000

num_chunks = 1

# filetered = x_rdf.Filter("m_jj < 1")

# print(x_rdf)
# print(filetered)

generator = Generator(x_rdf, columns, chunk_rows, batch_rows, num_chunks, use_whole_file=True)


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