#%%

import uproot
import numpy as np
import tensorflow as tf
import argparse

from BatchTimer import BatchTimer

import time

# from keras import backend as K
# jobs = 1
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=jobs, 
#                         inter_op_parallelism_threads=jobs, 
#                         allow_soft_placement=True, \
#                         device_count = {'CPU': jobs})
# session = tf.compat.v1.Session(config=config)
# K.set_session(session)

# %%


parser = argparse.ArgumentParser()

parser.add_argument("--num")

args = parser.parse_args()

num = args.num

main_folder = "/home/danteniewenhuis/Documents/CERN/TMVA-batching"


start = time.time()

file = uproot.open(f"{main_folder}/data/file_sizes_bench/{num}.root")
tree = file["test_tree"]
branches = tree.arrays()
target = "Type"

res = []

for k in tree.keys():
    if k == target:
        y = branches[k].to_numpy()
        continue
    res.append(branches[k].to_numpy())

data = np.array(res)
X = data.transpose()

print(X.shape)

loading_time =  time.time() - start
print(f"loading took: {loading_time}")

# %%

model = tf.keras.Sequential([
    tf.keras.layers.Dense(200, activation=tf.nn.tanh, input_shape=(28,)),  # input shape required
    tf.keras.layers.Dense(400, activation=tf.nn.tanh),
    tf.keras.layers.Dense(400, activation=tf.nn.tanh),
    tf.keras.layers.Dense(200, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x=X, y=y, batch_size=1024, validation_split=0.3, callbacks = [BatchTimer(f"Tensorflow_uproot_{num}", loading_time)], epochs=1)

# %%
