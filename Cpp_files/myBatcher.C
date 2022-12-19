#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

// Include ROOT files
#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

// Include my classes
#include "DataLoader.C"
#include "BatchGeneratorSpec.C"

#include <chrono>
#include <fstream>

void myBatcher()
{
    ofstream myFile;
    myFile.open("results/benchmark_DatasetSpec_100_000.txt");
    myFile << "0,";

    // Define variables
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                     "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                     "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                     "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                     "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"};
    size_t batch_size = 2000, chunk_size = 100000, max_chunks = 20000;

    // Load the RDataFrame and create a new tensor
    // ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "data/h5train_combined.root");
    // std::vector<std::string> cols = x_rdf.GetColumnNames();

    auto start = std::chrono::steady_clock::now();

    BatchGenerator generator("data/h5train_combined.root", "sig_tree", cols, chunk_size, batch_size, max_chunks);

    size_t i = 0;
    while(true) {
        auto batch = generator.get_batch();
        // std::cout << (*batch) << std::endl;
        i++;
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        myFile << elapsed_seconds.count() << ",";
        if (batch->GetSize() == 0) {
            break;
        }
    }

    myFile.close();

}

int main() {
    myBatcher();
}