import random
import numpy as np
import time


import multiprocessing as mp
from multiprocessing import shared_memory

import threading

def append_to_list(num_items):
    lst = []
    
    for n in random.sample(range(20_000_000), num_items):
        lst.append(n)

    return lst

def randomize_list(num_items, child_conn):
    print("randomize list")
    lst = child_conn.recv()
    
    for i, n in enumerate(random.sample(range(20_000_000), num_items)):
        lst[i] = n

    print("sending")
    child_conn.send(lst)
    child_conn.close()

def other_process(name, array_size):
    lst = shared_memory.ShareableList(name=name)
    
    for i, v in enumerate(random.sample(range(20_000_000), array_size)):
        lst[i] = v

    # print(f"{lst = }")

if __name__ == "__main__":
    array_size = 10_000_000

    thread = mp.Process(target=append_to_list, args=(array_size,))
    thread_2 = mp.Process(target=append_to_list, args=(array_size,))
    thread.start()
    thread_2.start()
    thread.join()
    thread_2.join()
    # lst = [0 for x in range(array_size)]
    # buff = shared_memory.ShareableList(lst)

    # process = mp.Process(target=other_process, args=(buff.shm.name, array_size))

    # process.start()

    # process.join()

    # # print(f"{buff = }")

    # buff.shm.close()
    # buff.shm.unlink()

    # array_size = 1_000_000

    # lst = [0 for x in range(array_size)]
    
    # parent_conn, child_conn = mp.Pipe()

    # print("created Pipe")

    # parent_conn.send(lst)

    # print("create process")
    # process = mp.Process(target=randomize_list, args=(array_size, child_conn))

    # process.start()
    # process.join

    # lst = parent_conn.recv()

