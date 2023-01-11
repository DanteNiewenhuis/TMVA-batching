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
size_t load_chunk(TMVA::Experimental::RTensor<float>& x_tensor, string file_name, string tree_name,
                std::vector<std::string> cols, const size_t num_columns, 
                const size_t chunk_rows, const size_t start_row = 0, bool random_order=true) 
{
    
    // Fill the RTensor with the data from the RDataFrame
""" + f"ChunkLoader<float, std::make_index_sequence<{num_columns}>>" + """
        func(x_tensor, num_columns, chunk_rows, random_order);

    long long start_l = start_row;
    long long end_l = start_l + chunk_rows;
    ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec(tree_name, 
                                                file_name, {start_l,  std::numeric_limits<Long64_t>::max()});
    ROOT::RDataFrame x_rdf = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);


    auto myCount = x_rdf.Range(0, chunk_rows).Count();
    x_rdf.Range(0, chunk_rows).Foreach(func, cols);
    return myCount.GetValue();
}
""")



class Generator:

    def __init__(self, file_name, tree_name, columns: list[str], chunk_rows: int, 
                 batch_rows: int, num_chuncks: int=1, use_whole_file: bool=True):
        
        # Initialize parameters
        self.file_name = file_name
        self.tree_name = tree_name
        self.columns = columns
        self.chunk_rows = chunk_rows
        self.EoF = False
        self.current_row = 0

        self.num_columns = len(columns)
        self.batch_rows = batch_rows
        self.batch_size = batch_rows * self.num_columns
        self.use_whole_file = use_whole_file

        load_functor(self.num_columns)

        f = ROOT.TFile.Open(file_name)
        t = f.Get(tree_name)
        self.entries = t.GetEntries()

        # Create x_tensor
        self.x_tensor = ROOT.TMVA.Experimental.RTensor("float")([self.chunk_rows, self.num_columns])    
        self.generator = ROOT.BatchLoader(self.batch_rows, self.num_columns)

    def load_chunk(self):
        print(f"{self.current_row = }")

        # Fill x_tensor and get the number of rows that were processed
        loaded_size = ROOT.load_chunk(self.x_tensor, self.file_name, self.tree_name, self.columns, self.num_columns, 
                                     self.chunk_rows, self.current_row, False)

        #TODO: think about what to do if end of file is reached
        if (loaded_size < self.batch_rows):
            self.EoF = True

        # Create Generator
        self.generator.SetTensor(self.x_tensor, loaded_size)

        self.current_row += self.chunk_rows

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
        if (self.current_row < self.entries):
            self.load_chunk()
            return self.__next__()

        raise StopIteration
