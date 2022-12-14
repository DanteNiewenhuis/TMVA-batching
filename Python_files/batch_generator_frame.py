import ROOT
import torch
import numpy as np

main_folder = "../"

def load_functor(num_columns):
    # Import myBatcher.C
    # ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/ChunkLoader.cpp"')
    ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/BatchGenerator.cpp"')

    # Create C++ function
    ROOT.gInterpreter.ProcessLine("""
size_t load_chunk(TMVA::Experimental::RTensor<float>& x_tensor, ROOT::RDF::RNode x_rdf,
                std::vector<std::string> cols, const size_t num_columns, 
                const size_t chunk_rows, const size_t start_row = 0, bool random_order=true) 
{
    
    // Fill the RTensor with the data from the RDataFrame
""" + f"ChunkLoader<float, std::make_index_sequence<{num_columns}>>" + """
        func(x_tensor, num_columns, chunk_rows, random_order);
    auto myCount = x_rdf.Range(start_row, start_row + chunk_rows).Count();
    x_rdf.Range(start_row, start_row + chunk_rows).Foreach(func, cols);
    return myCount.GetValue();
}
""")

class Generator:

    def __init__(self, x_rdf: ROOT.TMVA.Experimental.RTensor, columns: list[str], chunk_rows: int, 
                 batch_rows: int, num_chuncks: int=1, use_whole_file: bool=True):
        
        # Initialize parameters
        self.x_node = ROOT.RDF.AsRNode(x_rdf)
        self.columns = columns
        self.chunk_rows = chunk_rows
        self.current_chunck = 0
        self.num_chuncks = num_chuncks
        self.EoF = False

        self.num_columns = len(columns)
        self.batch_rows = batch_rows
        self.batch_size = batch_rows * self.num_columns
        self.use_whole_file = use_whole_file

        load_functor(self.num_columns)

        # Create x_tensor
        self.x_tensor = ROOT.TMVA.Experimental.RTensor("float")([self.chunk_rows, self.num_columns])    
        self.generator = ROOT.BatchLoader(self.batch_rows, self.num_columns)

    def load_chunk(self):
        start = self.current_chunck * self.chunk_rows
        print(f"{start = }")

        # Fill x_tensor and get the number of rows that were processed
        loaded_size = ROOT.load_chunk(self.x_tensor, self.x_node, self.columns, self.num_columns, 
                                     self.chunk_rows, start, False)

        #TODO: think about what to do if end of file is reached
        if (loaded_size < self.batch_rows):
            self.EoF = True

        # Create Generator
        self.generator.SetTensor(self.x_tensor, loaded_size)

    def __iter__(self):
        self.EoF = False
        self.current_chunck = 0
        self.load_chunk()

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
        if ((self.use_whole_file and not self.EoF) or (self.current_chunck < self.num_chuncks)):
            self.load_chunk()
            return self.__next__()

        raise StopIteration
