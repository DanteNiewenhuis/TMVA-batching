import ROOT
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten

def Batch_Generator(features, targets, batch_size=1):
    batch_count = 0
    features_batch = []
    targets_batch = []

    dataset_size = len(list(features.values())[0])

    for i in range(dataset_size):
        features_batch.append(np.array([features[x][i] for x in features.keys()]))
        targets_batch.append(np.array([targets[x][i] for x in targets.keys()]))
        if batch_count==batch_size-1:
            yield np.stack(features_batch), np.stack(targets_batch)
            features_batch = []
            targets_batch = []
            batch_count = 0
        else:
            batch_count += 1



df = ROOT.RDataFrame("sig_tree", "Higgs_data.root")
x_df = df.AsNumpy(columns=["jet1_phi","jet1_eta", "jet2_pt"])
y_df = df.AsNumpy(columns=["jet3_b-tag"])



batch_size = 4

print("end")

# train_generator = Batch_Generator(x_df, y_df, batch_size)


# model = Sequential()
# model.add(Dense(12, input_dim=3, activation="relu"))
# model.add(Dense(12, activation="relu"))
# model.add(Dense(1, activation="sigmoid"))
# model.summary()
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# model.fit(Batch_Generator(x_df, y_df, batch_size),steps_per_epoch=100/batch_size ,epochs=20)