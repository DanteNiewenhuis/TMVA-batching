import numpy as np
import ROOT

data_root = []
data_all = []

bins = 20
h_root = ROOT.TH1F("root", "root", bins, 0, 0.2)
h_all = ROOT.TH1F("all", "all", bins, 0, 0.2)


with open("results/Tensorflow_ROOT_5_100000_train.csv", "r") as rf:
    for 


for x, y in zip(data_root, data_all):
    h_root.Fill(x)
    h_all.Fill(y)

h_root_max = h_root.GetMaximum()
h_all_max = h_all.GetMaximum()

max_y = max(h_root_max, h_all_max)

c = ROOT.TCanvas("canvas", "canvas", 800, 800)

h_root.GetYaxis().SetRangeUser(0, max_y + 50)


h_root.SetFillColorAlpha(8, 0.5)
h_all.SetFillColorAlpha(9, 0.5)

h_root.Draw()
h_all.Draw("SAME")

c.Draw()
