from threading import Thread
import ROOT
import torch
import numpy as np
import time

main_folder = "../"

def load_functor(num_columns):
    # Import myBatcher.C
    ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/DataLoader.C"')
    ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/BatchGenerator.C"')

    # Create C++ function
    ROOT.gInterpreter.ProcessLine("""
size_t load_data(TMVA::Experimental::RTensor<float>& x_tensor, ROOT::RDF::RNode x_rdf,
                std::vector<std::string> cols, const size_t num_columns, 
                const size_t chunk_rows, const size_t start_row = 0, bool random_order=true) 
{
    
    // Fill the RTensor with the data from the RDataFrame
""" + f"DataLoader<float, std::make_index_sequence<{num_columns}>>" + """
        func(x_tensor, num_columns, chunk_rows, random_order);
    auto myCount = x_rdf.Range(start_row, start_row + chunk_rows).Count();
    x_rdf.Range(start_row, start_row + chunk_rows).Foreach(func, cols);
    return myCount.GetValue();
}
""")

class Generator:

    def __init__(self, x_rdf: ROOT.TMVA.Experimental.RTensor, columns: list[str], chunk_rows: int, 
                 batch_rows: int, num_chunks: int=1, use_whole_file: bool=True):
        
        # Initialize parameters
        self.x_node = ROOT.RDF.AsRNode(x_rdf)
        self.columns = columns
        self.num_columns = len(columns)

        self.chunk_rows = chunk_rows
        self.chunks_loaded = 0
        self.num_chunks = num_chunks
        self.use_whole_file = use_whole_file

        self.batch_rows = batch_rows
        self.batch_size = batch_rows * self.num_columns

        # Compile C++ function
        load_functor(self.num_columns)
        
        # Create two x_tensors and a generator
        self.x_tensors = [ROOT.TMVA.Experimental.RTensor("float")([self.chunk_rows, self.num_columns]) for _ in range(2)]
        self.tensor_length = [0, 0]
        self.current_tensor_idx = 0

        self.generator = ROOT.BatchGeneratorHelper(self.batch_rows, self.num_columns)
        self.EoF = False

    def load_chunk(self, tensor_idx: int):

        if (self.EoF):
            print("load_chunk => End of File")
            return

        start = self.chunks_loaded * self.chunk_rows
        print(f"load_chunk => Loading new data: {self.chunks_loaded = }")

        # Fill tensor_idx and get the number of rows that were processed
        self.tensor_length[tensor_idx] = ROOT.load_data(self.x_tensors[tensor_idx], self.x_node, self.columns, 
                                                        self.num_columns, self.chunk_rows, start, False)

        print(f"load_chunk => Done loading: {self.chunks_loaded = }")
        self.chunks_loaded += 1

        if self.tensor_length[tensor_idx] < self.chunk_rows:
            self.EoF = True

    def next_chunk(self):
        print("next chunk")
        
        self.thread.join()
        next_tensor_idx = abs(1-self.current_tensor_idx)
        # set the next tensor on the generator
        self.generator.SetTensor(self.x_tensors[next_tensor_idx], self.tensor_length[next_tensor_idx])
        
        # check if more chuncks need to be loaded
        if (self.chunks_loaded >= self.num_chunks):
            self.EoF = True
            return

        # load data on the next tensor TODO: make parallel
        self.thread = Thread(target=self.load_chunk, args=(self.current_tensor_idx,))
        self.thread.start()

        self.current_tensor_idx = next_tensor_idx


    def __iter__(self):
        # Load the first chunk into the current_tensor
        self.load_chunk(self.current_tensor_idx)
        self.generator.SetTensor(self.x_tensors[self.current_tensor_idx], 
                                 self.tensor_length[self.current_tensor_idx])

        # Load the first chunk into the next_tensor TODO: make parallel
        next_tensor_idx = abs(1-self.current_tensor_idx)
        self.thread = Thread(target=self.load_chunk, args=(next_tensor_idx,))
        self.thread.start()

        return self

    # Return a batch when available
    def __next__(self) -> ROOT.TMVA.Experimental.RTensor:
        
        if (self.generator.HasData()):
            batch = self.generator()
            data = batch.GetData()
            data.reshape((self.batch_size,))
            return torch.Tensor(data).view(self.batch_rows, self.num_columns)

        if self.EoF:
            print(f"Stop Iteration")
            raise StopIteration

        # Load the next chunk
        self.next_chunk()
        return self.__next__()

