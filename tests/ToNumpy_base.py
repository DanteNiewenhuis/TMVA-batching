import time
import numpy as np
import ROOT

ROOT.EnableImplicitMT()
ROOT.EnableThreadSafety()

start = time.perf_counter()
tree_name = "test_tree"
file_name = "../data/small_data.root"

x_rdf = ROOT.RDataFrame(tree_name, file_name)
end = time.perf_counter()

print(f"Getting rdf took {end - start}")


print(f"{x_rdf.GetColumnNames()}")

start = time.perf_counter()
x_array = x_rdf.AsNumpy()
end = time.perf_counter()

print(f"ToNumpy took {end - start}")


print(f"{x_array}")
