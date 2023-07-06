import subprocess
import re

main_folder = "/home/dante/Documents/TMVA-batching"

def get_performance_ROOT(num, chunksize):
    print("testing ROOT")
    output = subprocess.getstatusoutput(f"/usr/bin/time  python3 {main_folder}/benchmarks/memory_test/ROOT_simple_TF.py --num {num} --chunksize {chunksize}")

    if output[0] != 0:
        print(output[1])

        return

    memory_usage = int(re.findall(" (\d+)maxresident", output[1])[0])

    with open(f"{main_folder}/benchmarks/memory_test/results/memory.csv", "a") as wf:
        wf.write(f"ROOT,{num},{memory_usage},{chunksize}\n")

def get_performance_uproot(num):
    print("testing uproot")
    output = subprocess.getstatusoutput(f"/usr/bin/time  python3 {main_folder}/benchmarks/memory_test/uproot_simple_TF.py --num {num}")

    if output[0] != 0:
        print(output[1])

        return

    memory_usage = int(re.findall(" (\d+)maxresident", output[1])[0])

    with open(f"{main_folder}/benchmarks/memory_test/results/memory.csv", "a") as wf:
        wf.write(f"uproot,{num},{memory_usage}\n")


chunksize = 10_000_000
for num in range(5,14):
    if num == 10:
        continue
    print(f"processing file {chunksize = } {num = }")

    get_performance_ROOT(num, chunksize)
    

chunksize = 20_000_000
for num in range(10,14):
    if num == 10:
        continue
    print(f"processing file {chunksize = } {num = }")

    get_performance_ROOT(num, chunksize)
    