from ROOT.TMVA.Experimental import GetTFDatasets, GetBatchGenerators, GetPyTorchDataLoaders 

main_folder = "/home/dante/Documents/TMVA-batching"

tree_name = "test_tree"
file_name = f"{main_folder}/data/file_sizes_bench/{num}.root"
target = "Type"

1
batch_rows = 1024
chunk_rows = 100




# Returns a generator that returns NumPy arrays
gen_train, gen_validation = GetBatchGenerators(file_name, tree_name, chunk_rows,
                            batch_rows, target="Type", validation_split=0.3)

# Returns a TensorFlow Dataset
ds_train, ds_validation = GetTFDatasets(
                            file_name, tree_name, 
                            chunk_rows,batch_rows, 
                            target="Type", 
                            validation_split=0.3)

# Returns a generator that returns PyTorch Tensors
gen_train, gen_validation = GetPyTorchDataLoaders(file_name, tree_name, chunk_rows,
                            batch_rows, target="Type", validation_split=0.3)