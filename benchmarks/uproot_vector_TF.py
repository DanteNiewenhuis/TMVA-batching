#%%

import uproot
import numpy as np
import tensorflow as tf

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

start = time.time()

file = uproot.open("../data/vectorData.root")
tree = file["tree"]
branches = tree.arrays()
target = "labels"

res = []

vec_size = 100

for k in tree.keys():
    if k == target:
        y = branches[k].to_numpy()
        continue
    
    np_data = branches[k].to_numpy()

    if len(np_data.shape) > 1:
        for i in range(vec_size):
            res.append(np_data[:,i])
        continue

    res.append(np_data)

data = np.array(res)
X = data.transpose()

print(X.shape)

loading_time =  time.time() - start
print(f"loading took: {loading_time}")

# %%

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation=tf.nn.tanh, input_shape=(X.shape[1],)),  # input shape required
    tf.keras.layers.Dense(1000, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1000, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x=X, y=y, batch_size=1024, validation_split=0.3, callbacks = [BatchTimer("Tensorflow_vector_uproot", loading_time)], epochs=2)

# %%
