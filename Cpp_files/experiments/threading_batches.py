from threading import Thread
import time

class Generator:
    def __init__(self):
        
        self.current_tensor_idx = 0

        self.tensor_values = [0, 0]

        self.generator = 0

        self.counter = 0

        self.thread_started = False

        self.data_available = False

    def load_chunk(self, tensor_idx):
        print(f"CLASS: loading chunk: {tensor_idx = }, {self.counter = }")
        time.sleep(2)

        self.tensor_values[tensor_idx] = self.counter

        self.counter += 1

        print(f"CLASS: done loading chunk: {tensor_idx = }")

    def next_chunk(self):
        print("getting next chunk")

        next_tensor_idx = abs(1-self.current_tensor_idx)
        
        # set the next tensor on the generator
        self.thread.join()
        self.generator = self.tensor_values[next_tensor_idx]
        self.data_available = True

        self.thread = Thread(target=self.load_chunk, args=(self.current_tensor_idx,))
        self.thread.start()
        self.current_tensor_idx = next_tensor_idx

    def __iter__(self):
        self.load_chunk(self.current_tensor_idx)
        self.generator = self.tensor_values[self.current_tensor_idx]

        next_tensor_idx = abs(1-self.current_tensor_idx)
        self.thread = Thread(target=self.load_chunk, args=(next_tensor_idx,))
        self.thread.start()

        self.data_available = True
        self.thread_started = False
        return self
    
    def __next__(self):
        if self.counter > 3:
            raise StopIteration
        
        if self.data_available:
            self.data_available = False
            return self.generator

        self.next_chunk()
        return self.__next__()

if __name__ == '__main__':
    generator = Generator()

    for batch in generator:
        print(f"MAIN: {batch = }")
        time.sleep(4)
        print("MAIN: batch processed")
