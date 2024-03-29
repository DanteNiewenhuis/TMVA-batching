import ROOT

ROOT.EnableThreadSafety()


# from ROOT.TMVA.Experimental import GetGenerators, CreateTFDatasets
import time
import numpy as np
import argparse

import tensorflow as tf

main_folder = "../"

tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

# tree_name = "sig_tree"
# file_name = f"{main_folder}data/vectorData.root"

batch_rows = 1024
chunk_rows = 1_000_000

# columns = ['jet1_btag', 'jet1_eta', 'Type']
# columns = []

ds_train, ds_validation = ROOT.TMVA.Experimental.CreateTFDatasets(
    file_name, tree_name, chunk_rows, batch_rows, target="Type", validation_split=0.3
)


# gen_train, gen_validation = GetGenerators(file_name, tree_name, chunk_rows,
#                            batch_rows, target="Type", validation_split=0.3, max_chunks=2)


# for item in ds_train:
#     print(f"train: {type(item)}, {item}")

# for item in ds_validation:
#     print(f"validation: {type(item)}, {item}")

# num_columns = len(train_generator.columns)

# ###################################################################################################
# ## AI example
# ###################################################################################################

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            300, activation=tf.nn.tanh, input_shape=(28,)
        ),  # input shape required
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])


model.fit(ds_train, validation_data=ds_validation, epochs=3)
