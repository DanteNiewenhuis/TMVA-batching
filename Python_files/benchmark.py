from batch_generator_parralel import Generator
import ROOT
import numpy
import time

import matplotlib.pyplot as plt

main_folder = "../"

# columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", 
#         "fjet_ECF2", "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
#         "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", 
#         "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", 
#         "fjet_pt", "weights"]

# columns = ["fjet_C2", "fjet_D2", "fjet_ECF1"]

# columns = ["m_jj", "m_jjj", "m_jlv"] 
# x_rdf = ROOT.RDataFrame("sig_tree", f"{main_folder}data/Higgs_data_full.root", columns)
# x_rdf = ROOT.RDataFrame("sig_tree", f"{main_folder}data/h5test_combined.root", columns)

columns = ["m_jj", "m_jjj", "m_jlv"] 
x_rdf = ROOT.RDataFrame("sig_tree", f"{main_folder}data/Higgs_data_full.root", columns)

num_columns = len(columns)
batch_rows = 2000
chunk_rows = 1_000_000

start = time.time()
generator = Generator(x_rdf, columns, chunk_rows, batch_rows, use_whole_file=True)

middle = time.time()

print(f"loading took: {middle - start}")

times = [0]

# i = 0
# goal = 20_000
# while i < goal:
#     for batch in enumerate(generator):
#         times.append(time.time() - middle)
        
#         if (i % 50 == 0):
#             print(f"batch {i}")

#         if i >= goal:
#             break
        
#         i += 1

#     print("loop done")

for j in range(2):
    for i, batch in enumerate(generator):
        times.append(time.time() - middle)

end = time.time()
print(f"batching took: {end - middle}")

with open(f"{main_folder}results/data.txt", "w") as f:
    f.write(",".join([str(i) for i in times]))

# plt.plot(times)
# plt.savefig(f"{main_folder}results/test.png")

# plt.show()