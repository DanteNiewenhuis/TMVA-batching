import ROOT

ROOT.EnableThreadSafety()

from BatchTimer import BatchTimer

# from ROOT.TMVA.Experimental import GetGenerators, CreateTFDatasets
import time
import numpy as np
import argparse

import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument("--num", type=int)
parser.add_argument("--chunksize", type=int)

args = parser.parse_args()

num = args.num

main_folder = "/home/dante/Documents/TMVA-batching"

tree_name = "test_tree"
file_name = f"{main_folder}/data/file_sizes_bench/{num}.root"

batch_rows = 1024
chunk_rows = int(args.chunksize)


ds_train, ds_validation = ROOT.TMVA.Experimental.CreateTFDatasets(
    file_name, tree_name, chunk_rows, batch_rows, target="Type", validation_split=0.3
)


# raise NotImplementedError

# ###################################################################################################
# ## AI example
# ###################################################################################################

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            200, activation=tf.nn.tanh, input_shape=(28,)
        ),  # input shape required
        tf.keras.layers.Dense(400, activation=tf.nn.tanh),
        tf.keras.layers.Dense(400, activation=tf.nn.tanh),
        tf.keras.layers.Dense(200, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])


model.fit(
    ds_train,
    validation_data=ds_validation,
    epochs=1,
    callbacks=[BatchTimer(f"Tensorflow_ROOT_{num}_{chunk_rows}")],
)
