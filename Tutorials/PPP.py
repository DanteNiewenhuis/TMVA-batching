import ROOT

ROOT.EnableThreadSafety()

from BatchTimer import BatchTimer

# from ROOT.TMVA.Experimental import GetGenerators, CreateTFDatasets
import time
import numpy as np
import argparse

import tensorflow as tf


tree_name = "test_tree"
file_name = f"{main_folder}/data/Higgs_data_full.root"

batch_size = 1024
chunk_size = 1_000_000
target = "Type"
weight = "W"

filters = []

data_X = np.array(...)
data_y = np.array(...)

model = tf.keras.Sequential(...)
loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(x=data_X, y=data_y, batch_size=batch_size, validation_split=0.3, epochs=2)


ds_train, ds_validation = ROOT.TMVA.Experimental.CreateTFDatasets(
    tree_name, file_name, batch_size, chunk_size, target=target, validation_split=0.3
)

model = tf.keras.Sequential(...)
loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(ds_train, validation_data=ds_validation, epochs=2)


gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    tree_name,
    file_name,
    batch_size,
    chunk_size,
    target=target,
    weight=weight,
    validation_split=0.3,
    filters=filters,
    shuffle=False,
)

for x, y in gen_train:
    print(f"Input: {x}, target: {y}")

# Validation
for x, y in gen_validation:
    print(f"Input: {x}, target: {y}")
