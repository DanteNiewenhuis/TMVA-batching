from threading import Thread
import ROOT
import torch
import numpy as np
import time

main_folder = "../"

def load_functor(file_name, tree_name, columns) -> list:
    # Import myBatcher.C
    ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/ChunkLoader.cpp"')
    ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/BatchLoader.cpp"')

    x_rdf = ROOT.RDataFrame(tree_name, file_name, columns)
    columns = list(x_rdf.GetColumnNames())

    template_dict = {"Int_t": "int&", "Float_t": "float&"}

    template_string = ""
    for name in columns:
        template_string += template_dict[x_rdf.GetColumnType(name)] + ","


    # Create C++ function
    ROOT.gInterpreter.ProcessLine("""
std::tuple<size_t, size_t> load_chunk(TMVA::Experimental::RTensor<float>& x_tensor, string file_name, string tree_name, std::vector<std::string> filters,
                std::vector<std::string> cols, const size_t num_columns, 
                const size_t chunk_size, const size_t current_row = 0, bool random_order=true) 
{
    
    // Fill the RTensor with the data from the RDataFrame
""" + f"ChunkLoader<{template_string[:-1]}>" + """
        func(x_tensor);

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

    return columns

class Generator:

    def __init__(self, file_name, tree_name, columns: list[str], filters: list[str], chunk_rows: int, 
                 batch_rows: int, target: str = "", num_chuncks: int=1):
        
        # Initialize parameters
        self.file_name = file_name
        self.tree_name = tree_name
        self.filters = filters
        self.chunk_rows = chunk_rows
        self.current_row = 0

        f = ROOT.TFile.Open(file_name)
        t = f.Get(tree_name)
        self.entries = t.GetEntries()

        # Load C++ functions
        start = time.time()
        self.columns = load_functor(self.file_name, self.tree_name, columns)
        print(f"loading functor took: {time.time() - start}")


        self.num_columns = len(columns)
        self.batch_rows = batch_rows
        self.batch_size = batch_rows * self.num_columns

        self.target_index = self.columns.index(target)
        
        # Create two x_tensors and a generator
        self.x_tensors = [ROOT.TMVA.Experimental.RTensor("float")([self.chunk_rows, self.num_columns]) for _ in range(2)]
        self.tensor_length = [0, 0]
        self.training_tensor = 0
        self.loading_tensor = 1

        self.generator = ROOT.BatchLoader(self.batch_rows, self.num_columns)
        self.EoF = False

    # LOAD
    def load_chunk(self, tensor_idx: int):  
        print(f"load_chunk => {self.current_row = }")
        time.sleep(2)
        print(f"Done sleep")

        # Fill tensor_idx and get the number of rows that were processed
        progressed_size, loaded_size = ROOT.load_chunk(self.x_tensors[tensor_idx], self.file_name, self.tree_name, self.filters,
                                     self.columns, self.num_columns, self.chunk_rows, self.current_row, False)

        print("done loading")
        # print(f"load_chunk => Done loading: {self.chunks_loaded = }")
        self.current_row += progressed_size
        self.tensor_length[tensor_idx] = loaded_size

    # SET
    def set_tensor(self, idx: int):
        self.generator.SetTensor(self.x_tensors[idx], 
                            self.tensor_length[idx])

    # SWAP
    def swap_tensors(self):
        temp = self.training_tensor
        self.training_tensor = self.loading_tensor
        self.loading_tensor = temp

    # Start
    def __iter__(self):
        self.EoF = False
        self.chunks_loaded = 0
        self.training_tensor = 0
        self.loading_tensor = 1

        # Load T_training
        self.load_chunk(self.training_tensor)

        # Load T_loading
        self.thread = Thread(target=self.load_chunk, args=(self.loading_tensor,))
        self.thread.start()

        # Set T_training
        self.set_tensor(self.training_tensor)

        # ML T_training
        return self

    # Middle
    def next_chunk(self):
        print(f"next_chunk {self.current_row = }")
        # wait untill the loading tensor is loaded
        start = time.time()
        self.thread.join()
        end = time.time()
        print(f"joining took: {end - start}")

        # Swap Tensors
        start = time.time()
        self.swap_tensors()
        end = time.time()
        print(f"swapping took: {end - start}")

        # Set T_training
        start = time.time()
        self.set_tensor(self.training_tensor)
        end = time.time()
        print(f"setting took: {end - start}")

        # Load T_loading if EoF is not reached yet
        if self.current_row < self.entries:
            start = time.time()
            self.thread = Thread(target=self.load_chunk, args=(self.loading_tensor,))
            self.thread.start()
            end = time.time()

            print(f"starting took: {end - start}")

        else:
            self.EoF = True

    # Return a batch when available
    def __next__(self) -> ROOT.TMVA.Experimental.RTensor:
        
        if (self.generator.HasData()):
            time.sleep(0.5)
            batch = self.generator()
            data = batch.GetData()
            data.reshape((self.batch_size,))
            return_data = np.array(data).reshape(self.batch_rows, self.num_columns)

            return np.column_stack((return_data[:,:self.target_index], return_data[:,self.target_index+1:])), \
                   return_data[:,self.target_index]

        if self.EoF:
            print(f"Stop Iteration")
            raise StopIteration

        # Load the next chunk
        start = time.time()
        self.next_chunk()
        end = time.time()
        print(f"getting next took: {end - start}\n")
        return self.__next__()

