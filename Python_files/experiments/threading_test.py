import time
from threading import Thread

### Single
def load_tensor_single(i):
    print(f"Loading tensor {i}")
    time.sleep(0.5)
    print(f"Loaded tensor {i}")

def process_single(i):
    print(f"Processing tensor {i}")
    time.sleep(0.5)
    print(f"Processed tensor {i}")

def single(loops):
    load_tensor_single(0)
    for i in range(loops-1):
        process_single(i)
        load_tensor_single(i+1)
    
    process_single(i+1)

### parralel 
thread = Thread()

def load_tensor_parralel(i):
    print(f"Loading tensor {i}")
    time.sleep(0.5)
    print(f"Loaded tensor {i}")

def process_parralel(i):
    print(f"Processing tensor {i}")
    time.sleep(0.5)
    print(f"Processed tensor {i}")

def parralel(loops):
    thread = Thread(target=load_tensor_parralel, args=(0,))
    thread.start()
    for i in range(loops-1):
        thread.join()
        thread = Thread(target=load_tensor_parralel, args=(i+1,))
        thread.start()
        process_parralel(i)
    
    process_parralel(i+1)

start_single = time.time()
single(5)
end_single = time.time()

start_parralel = time.time()
parralel(5)
end_parralel = time.time()

print(f"single: {end_single - start_single}")
print(f"parralel: {end_parralel - start_parralel}")