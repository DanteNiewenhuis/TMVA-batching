import time
from tensorflow import keras

main_folder = "/home/dante/Documents/TMVA-batching"

class BatchTimer(keras.callbacks.Callback):
    def __init__(self, file_name, loading_time=0):
        super(BatchTimer, self).__init__()

        self.file_name = file_name

        self.train_timings = [loading_time]
        self.test_timings = []

    def on_train_batch_begin(self, batch, logs=None):
        self.start_time=time.time()    

    def on_train_batch_end(self, batch, logs=None):
        stop_time=time.time()

        self.train_timings.append(stop_time-self.start_time) 

    def on_train_end(self, logs=None):
        with open(f"{main_folder}/benchmarks/memory_test/results/{self.file_name}_train.csv", "w") as wf:
            for timing in self.train_timings:
                wf.write(f"{timing}\n")
        
    def on_test_batch_begin(self, batch, logs=None):
        self.start_time=time.time()    

    def on_test_batch_end(self, batch, logs=None):
        self.test_timings.append(time.time()-self.start_time) 

    def on_test_end(self, logs=None):
        with open(f"{main_folder}/benchmarks/memory_test/results/{self.file_name}_test.csv", "w") as wf:
            for timing in self.test_timings:
                wf.write(f"{timing}\n")