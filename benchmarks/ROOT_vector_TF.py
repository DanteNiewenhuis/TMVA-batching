import ROOT
ROOT.EnableThreadSafety()

from BatchTimer import BatchTimer

# from ROOT.TMVA.Experimental import GetGenerators, GetTFDatasets
import time
import numpy as np
import argparse

import tensorflow as tf

main_folder = "../"

tree_name = "tree"
file_name = f"{main_folder}data/vectorData.root"

batch_rows = 1024
chunk_rows = 100_000

vec_sizes = [100,100,100,100]

ds_train, ds_validation = ROOT.TMVA.Experimental.GetTFDatasets(file_name, tree_name, chunk_rows,
                           batch_rows, target="labels", vec_sizes=vec_sizes, validation_split=0.3)


# ###################################################################################################
# ## AI example
# ###################################################################################################

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation=tf.nn.tanh, input_shape=(sum(vec_sizes) + 20,)),  # input shape required
    tf.keras.layers.Dense(1000, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1000, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(ds_train, validation_data=ds_validation, epochs=2, callbacks = [BatchTimer("Tensorflow_vector_ROOT")])