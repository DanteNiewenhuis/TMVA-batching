import tensorflow as tf
import ROOT

ROOT.EnableThreadSafety()

main_folder = "../"


tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"

batch_rows = 1024
chunk_rows = 1_000_000

ds_train, ds_valid = ROOT.TMVA.Experimental.CreateTFDatasets(
    tree_name,
    file_name,
    batch_rows,
    chunk_rows,
    validation_split=0.3,
    target="Type",
)

num_train_columns = len(ds_train.train_columns)
num_target_columns = len(ds_train.target_columns)

print(ds_train.train_columns)

###################################################################################################
# AI example
###################################################################################################

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(300, activation=tf.nn.tanh, input_shape=(num_train_columns,)),  # input shape required
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(num_target_columns, activation=tf.nn.sigmoid),
    ]
)

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])


model.fit(ds_train, validation_data=ds_valid, epochs=2)
