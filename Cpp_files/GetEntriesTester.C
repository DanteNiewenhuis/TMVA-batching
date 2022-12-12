#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TFile.h"

void GetEntriesTester() {
    TFile* f = TFile::Open("data/r0-20.root");

    TTree* t = f->Get<TTree>("sig_tree");

    size_t entries = t->GetEntries();

    std::cout << entries << std::endl;
}