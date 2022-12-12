#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

// Include ROOT files
#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

// Include my classes
#include "DataLoader.C"
#include "BatchGenerator.C"

#include <chrono>

void myBatcher()
{
    // Define variables
    // std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv", "m_lv"};
    size_t batch_size = 2000, chunk_size = 1000001, max_chunks = 20000;

    // Load the RDataFrame and create a new tensor
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("test_tree", "data/Higgs_data_full.root");
    std::vector<std::string> cols = x_rdf.GetColumnNames();


    BatchGenerator generator(x_rdf, cols, chunk_size, batch_size, max_chunks);
    
    auto start = std::chrono::steady_clock::now();
    size_t i = 0;
    while(true) {
        auto batch = generator.get_batch();
        // if (i % 50 == 0) {

        //     std::cout << i << std::endl;
        // }
        i++;

        if (batch->GetSize() == 0) {
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

}

int main() {
    myBatcher();
}