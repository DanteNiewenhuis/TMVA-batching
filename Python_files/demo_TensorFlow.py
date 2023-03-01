import ROOT
ROOT.EnableThreadSafety()


from batch_generator import GetGenerators, GetTFDatasets
import tensorflow as tf

from BatchTimer import BatchTimer

from keras import backend as K
import tensorflow as tf

jobs =1

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=jobs,
                        inter_op_parallelism_threads=jobs,
                        allow_soft_placement=True,
                        device_count = {'CPU': jobs})
session = tf.compat.v1.Session(config=config)
K.set_session(session)

main_folder = "../"


tree_name = "test_tree"
file_name = f"{main_folder}data/Higgs_data_full.root"


# tree_name = "sig_tree"
# file_name = f"{main_folder}data/h5train_combined.root"


# columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
#                                 "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
#                                 "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
#                                 "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
#                                 "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights", "labels"]
                                
filters = ["fjet_D2 < 5"] # Random filters as example
# filters = []

batch_rows = 1024
chunk_rows = 1_000_000

ds_train, ds_valid = GetTFDatasets(file_name, tree_name, chunk_rows,
                           batch_rows, validation_split=0.3, target="Type")


###################################################################################################
## AI example
###################################################################################################

model = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation=tf.nn.tanh, input_shape=(28,)),  # input shape required
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(ds_train, validation_data=ds_valid, epochs=1, callbacks = [BatchTimer("Tensorflow_ROOT")])
