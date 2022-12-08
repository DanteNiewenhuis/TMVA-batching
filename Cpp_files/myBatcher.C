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

void myBatcher()
{
    // Define variables
    std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv", "m_lv"};
    size_t batch_size = 10, start_row = 0, chunk_size = 10, max_chunks = 20, num_columns = cols.size();

    // Load the RDataFrame and create a new tensor
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "data/r0-20.root", cols);

    BatchGenerator generator(x_rdf, cols, chunk_size, batch_size, max_chunks);

    size_t i = 0;
    while(true) {
        auto batch = generator.get_batch();
        std::cout << i++ << std::endl;

        if (batch->GetSize() == 0) {
            break;
        }
    }
}

int main() {
    myBatcher();
}