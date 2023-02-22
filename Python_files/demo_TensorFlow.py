import ROOT
from batch_generator import GetGenerators
import tensorflow as tf

from BatchTimer import BatchTimer

main_folder = "../"


tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"


# columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
#                                 "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
#                                 "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
#                                 "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
#                                 "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights", "labels"]
                                
filters = ["fjet_D2 < 5"] # Random filters as example
# filters = []

batch_rows = 1024
chunk_rows = 200_000

train_generator, test_generator = GetGenerators(file_name, tree_name, chunk_rows, batch_rows, target="Type", 
                                                train_ratio=0.7, use_whole_file= False, max_chunks=1)

num_columns = len(train_generator.columns)

###################################################################################################
## AI example
###################################################################################################

model = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation=tf.nn.tanh, input_shape=(num_columns-1,)),  # input shape required
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(train_generator, validation_data=test_generator, epochs=1, callbacks = [BatchTimer("Tensorflow_ROOT")])
