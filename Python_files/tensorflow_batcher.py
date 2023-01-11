import ROOT
import tensorflow as tf
import numpy as np
import time

main_folder = "../"

ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/ChunkLoader.cpp"')
# ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/BatchGenerator.cpp"')

tree_name = "sig_tree"
file_name = f"{main_folder}data/h5train_combined.root"


x_rdf = ROOT.RDataFrame(tree_name, file_name)
columns = x_rdf.GetColumnNames()

columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"]


chunk_rows = 100_000
batch_rows = 2_000
num_columns = len(columns)
chunk_size = chunk_rows*num_columns

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

class TFGenerator:

    def __init__(self, file_name, tree_name, columns: list[str], chunk_rows: int, 
                 batch_rows: int):

        self.file_name = file_name
        self.tree_name = tree_name
        self.columns = columns
        self.chunk_rows = chunk_rows
        self.EoF = False
        self.current_row = 0

        self.num_columns = len(columns)
        self.chunk_size = chunk_rows * self.num_columns

        self.x_tensor = ROOT.TMVA.Experimental.RTensor("float")([self.chunk_rows, self.num_columns]) 

        self.current_row = 0

    def load_chunk(self):
        if self.EoF:
            return []

        print(self.current_row)
        loaded_size = ROOT.load_chunk(self.x_tensor, self.file_name, self.tree_name, self.columns, self.num_columns, 
                                        self.chunk_rows, self.current_row, False)

        if (loaded_size < self.chunk_rows):
            self.EoF = True

        data = self.x_tensor.GetData()
        data.reshape((self.chunk_size,))
        data = np.array(data).reshape(self.chunk_rows, self.num_columns)

        self.current_row += self.chunk_rows

        return data

start = time.time()

generator = TFGenerator(file_name, tree_name, columns, chunk_rows, batch_rows)

middle = time.time()

times = [0]
while True:
    chunk = generator.load_chunk()
    if len(chunk) < chunk_rows:
        break

    ds = tf.data.Dataset.from_tensor_slices(chunk)
    
    ds_batched = ds.shuffle(len(ds)).batch(batch_size=batch_rows)

    for batch in ds_batched:
        times.append(time.time() - middle)

with open(f"{main_folder}results/Cpp_Python/{chunk_rows}_TF.txt", "w") as f:
    f.write("\n".join([str(i) for i in times]))
