import ROOT
ROOT.EnableThreadSafety()


from batch_generator import GetGenerators, GetTFDatasets
import time
import numpy as np
import argparse

import tensorflow as tf

main_folder = "../"

tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

batch_rows = 1024
chunk_rows = 150_000

ds_train, ds_validation = GetTFDatasets(file_name, tree_name, chunk_rows,
                           batch_rows, validation_split=0.3, target="Type", weights="m_wbb", max_chunks=2)



# num_columns = len(train_generator.columns)

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


model.fit(ds_train, validation_data=ds_validation, epochs=3)