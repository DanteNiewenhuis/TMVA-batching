import ROOT
import torch
import numpy as np
import time

main_folder = "../"


tree_name = "sig_tree"
file_name = f"{main_folder}data/h5train_combined.root"

x_rdf = ROOT.RDataFrame(tree_name, file_name)
columns = x_rdf.GetColumnNames()

columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"]

chunk_size = 1_000_000
batch_rows = 2_000
max_chunks = 200_000
num_columns = len(columns)
batch_size = batch_rows*num_columns


# ROOT.gInterpreter.ProcessLine(f'#include "{main_folder}Cpp_files/BatchGeneratorSpec.C"')
def test_Cpp():

    print("Testing Cpp")

    start = time.time()

    generator = ROOT.BatchGenerator(file_name, tree_name, columns, chunk_size, batch_rows, max_chunks)

    middle = time.time()

    print(f"loading took: {middle - start}")

    times = [0]
    i = 0
    while (generator.hasData()):
        batch = generator.get_batch()

        data = batch.GetData()
        data.reshape((batch_size,))
        data = torch.Tensor(data).view(batch_rows, num_columns)

        i += 1

        times.append(time.time() - middle)

    end = time.time()
    print(f"batching took: {end - middle}")

    with open(f"{main_folder}results/Cpp_Python/{chunk_size}_Python_Cpp_Spec.txt", "w") as f:
        f.write("\n".join([str(i) for i in times]))

from batch_generator_spec import Generator
def test_Python():

    print("Testing Python")

    start = time.time()

    generator = Generator(file_name, tree_name, columns, chunk_size, batch_rows, use_whole_file=True)

    # generator = Generator(x_rdf, columns, chunk_size, batch_rows, use_whole_file=True)

    middle = time.time()

    print(f"loading took: {middle - start}")

    times = [0]
    for batch in generator:
        times.append(time.time() - middle)

    end = time.time()
    print(f"batching took: {end - middle}")

    with open(f"{main_folder}results/Cpp_Python/{chunk_size}_Python_Spec.txt", "w") as f:
        f.write("\n".join([str(i) for i in times]))

test_Python()