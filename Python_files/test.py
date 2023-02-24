import ROOT
ROOT.EnableThreadSafety()


from batch_generator import GetGenerators
import time
import numpy as np
import argparse

# import tensorflow as tf

main_folder = "../"

tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

batch_rows = 1024
chunk_rows = 10_000

train_generator, validation_generator = GetGenerators(file_name, tree_name, chunk_rows,
                           batch_rows, validation_split=0.3, target="Type",
                           use_whole_file=False, max_chunks=2)

start = time.time()

train_generator.Activate()

end = time.time()


print(f"Main => Activation time: {end - start}")

train_generator.Activate()

train_generator.DeActivate()

# time.sleep(5)

# for item in ds_train:
#     end = time.time()
#     print(f"time: {end - start}")
#     start = end

# time.sleep(2)
# print("\nSECOND\n")

# start = time.time()

# for item in ds_train:
#     end = time.time()
#     print(f"time: {end - start}")
#     start = end



# num_columns = len(train_generator.columns)

# ###################################################################################################
# ## AI example
# ###################################################################################################

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(300, activation=tf.nn.tanh, input_shape=(num_columns-1,)),  # input shape required
#     tf.keras.layers.Dense(300, activation=tf.nn.tanh),
#     tf.keras.layers.Dense(300, activation=tf.nn.tanh),
#     tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])

# loss_fn = tf.keras.losses.BinaryCrossentropy()

# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])


# model.fit(ds_train, validation_data=ds_validation, epochs=2)