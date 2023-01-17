import ROOT
from batch_generator import Generator

main_folder = "../"

tree_name = "sig_tree"
file_name = f"{main_folder}data/test.root"

x_rdf = ROOT.RDataFrame(tree_name, file_name)

columns = x_rdf.GetColumnNames()

columns = ["weights", "fjet_C2", "labels"]
filters = []

chunk_rows = 10
batch_rows = 5

generator = Generator(file_name, tree_name, columns, filters, chunk_rows, batch_rows, target="labels")

for batch in generator:
    print(batch)