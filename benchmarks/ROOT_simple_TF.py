import ROOT
ROOT.EnableThreadSafety()

from BatchTimer import BatchTimer

# from ROOT.TMVA.Experimental import GetGenerators, GetTFDatasets
import time
import numpy as np
import argparse

import tensorflow as tf

main_folder = "../"

tree_name = "sig_tree"
file_name = f"{main_folder}data/Higgs_data_5.root"

batch_rows = 1024
chunk_rows = 1_000_000


ds_train, ds_validation = ROOT.TMVA.Experimental.GetTFDatasets(file_name, tree_name, chunk_rows,
                           batch_rows, target="jet1_btag", validation_split=0.3)


# raise NotImplementedError

# ###################################################################################################
# ## AI example
# ###################################################################################################

model = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation=tf.nn.tanh, input_shape=(27,)),  # input shape required
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(ds_train, validation_data=ds_validation, epochs=2, callbacks = [BatchTimer("Tensorflow_ROOT")])