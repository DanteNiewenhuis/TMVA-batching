#%%

import uproot
import numpy as np
import tensorflow as tf

from BatchTimer import BatchTimer

import time

# %%

start = time.time()

file = uproot.open("../data/Higgs_data_full.root")
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

print(f"loading took: {time.time() - start}")

# dataset_x = tf.data.Dataset.from_tensor_slices(X).batch(1024)
# dataset_y = tf.data.Dataset.from_tensor_slices(y).batch(1024)

# %%

model = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation=tf.nn.tanh, input_shape=(X.shape[1],)),  # input shape required
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x=X, y=y, batch_size=1024, validation_split=0.3, callbacks = [BatchTimer("Tensorflow_uproot")])

# %%
