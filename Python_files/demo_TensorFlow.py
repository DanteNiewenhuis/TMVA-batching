import ROOT
from batch_generator import Generator
import tensorflow as tf

main_folder = "../"


tree_name = "sig_tree"
file_name = f"{main_folder}data/h5train_combined.root"

# x_rdf = ROOT.RDataFrame(tree_name, file_name)
# columns = x_rdf.GetColumnNames()

columns = ["fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"]
                                
filters = ["fjet_D2 < 5"] # Random filter as example
# filters = []

num_columns = len(columns)
batch_rows = 1024
chunk_rows = 200_000

generator = Generator(file_name, tree_name, columns, filters, chunk_rows, batch_rows, use_whole_file=True)

###################################################################################################
## AI example
###################################################################################################

model = tf.keras.Sequential([
  tf.keras.layers.Dense(50, activation=tf.nn.relu, input_shape=(num_columns-1,)),  # input shape required
  tf.keras.layers.Dense(30, activation=tf.nn.relu),
  tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(generator)
