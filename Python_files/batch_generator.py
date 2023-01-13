import ROOT
import torch
import numpy as np

main_folder = "../"

def load_functor(num_columns):
    # Import myBatcher.C
    ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/ChunkLoader.cpp"')
    ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/BatchLoader.cpp"')

    # Create C++ function
    ROOT.gInterpreter.ProcessLine("""
std::tuple<size_t, size_t> load_chunk(TMVA::Experimental::RTensor<float>& x_tensor, string file_name, string tree_name, std::vector<std::string> filters,
                std::vector<std::string> cols, const size_t num_columns, 
                const size_t chunk_size, const size_t current_row = 0, bool random_order=true) 
{
    
    // Fill the RTensor with the data from the RDataFrame
""" + f"ChunkLoader<float, std::make_index_sequence<{num_columns}>>" + """
        func(x_tensor, num_columns, chunk_size, random_order);

    // Create DataFrame        
    long long start_l = current_row;
    long long end_l = start_l + chunk_size;
    ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec(tree_name, 
                                            file_name, {start_l, std::numeric_limits<Long64_t>::max()});
    ROOT::RDataFrame x_rdf = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);

    size_t progressed_events, passed_events;

    // add filters if given
    if (filters.size() > 0) {
        auto x_filter = x_rdf.Filter(filters[0], "F1");

        for (auto i = 1; i < filters.size(); i++) {
            auto name = "F" + std::to_string(i);
            x_filter = x_filter.Filter(filters[i], name);
        }

        // add range
        auto x_ranged = x_filter.Range(chunk_size);
        auto myReport = x_ranged.Report();

        // load data
        x_ranged.Foreach(func, cols);

        // get the loading info
        progressed_events = myReport.begin()->GetAll();
        passed_events = (myReport.end()-1)->GetPass();
    }
    
    // no filters given
    else {
        
        // add range
        auto x_ranged = x_rdf.Range(chunk_size);
        auto myCount = x_ranged.Count();

        // load data
        x_ranged.Foreach(func, cols);

        // get loading info
        progressed_events = myCount.GetValue();
        passed_events = myCount.GetValue();
    }

    return {progressed_events, passed_events}; 

}
""")


class Generator:

    def __init__(self, file_name, tree_name, columns: list[str], filters, chunk_rows: int, 
                 batch_rows: int, num_chuncks: int=1, use_whole_file: bool=True):
        
        # Initialize parameters
        self.file_name = file_name
        self.tree_name = tree_name
        self.columns = columns
        self.filters = filters
        self.chunk_rows = chunk_rows
        self.current_row = 0

        self.num_columns = len(columns)
        self.batch_rows = batch_rows
        self.batch_size = batch_rows * self.num_columns
        self.use_whole_file = use_whole_file

        # Load C++ functions
        load_functor(self.num_columns)

        f = ROOT.TFile.Open(file_name)
        t = f.Get(tree_name)
        self.entries = t.GetEntries()

        # Create x_tensor
        self.x_tensor = ROOT.TMVA.Experimental.RTensor("float")([self.chunk_rows, self.num_columns])    
        self.generator = ROOT.BatchLoader(self.batch_rows, self.num_columns)

    def load_chunk(self):
        # Fill x_tensor and get the number of rows that were processed
        progressed_size, loaded_size = ROOT.load_chunk(self.x_tensor, self.file_name, self.tree_name, self.filters,
                                     self.columns, self.num_columns, self.chunk_rows, self.current_row, False)

        # Create Generator
        self.generator.SetTensor(self.x_tensor, loaded_size)

        self.current_row += progressed_size

    def __iter__(self):
        self.current_chunck = 0
        self.load_chunk()

        return self

    # Return a batch when available
    def __next__(self) -> ROOT.TMVA.Experimental.RTensor:
        
        if (self.generator.HasData()):
            batch = self.generator()
            data = batch.GetData()
            data.reshape((self.batch_size,))
            return_data = np.array(data).reshape(self.batch_rows, self.num_columns)
            return return_data[:,:-1], return_data[:,-1]

        # Load the next chunk
        if (self.current_row < self.entries):
            self.load_chunk()
            return self.__next__()

        raise StopIteration
