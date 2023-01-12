#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

// Include ROOT files
#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

// Include my classes
#include "BatchGenerator.cpp"

#include <fstream>

void Test()
{
    // Define variables
    std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv"}; 
    // std::vector<std::string> filters = {"m_jj < 0.9", "m_jj > 0.4"};
    std::vector<std::string> filters = {};
    size_t batch_size = 1, chunk_size = 2, max_chunks = 20000;


    // auto file_name = "data/Higgs_data_full.root";
    auto file_name = "data/r0-10.root";
    auto tree_name = "sig_tree";

    BatchGenerator generator(file_name, tree_name, cols, filters, chunk_size, batch_size, max_chunks);

    while(generator.hasData()) {
        auto batch = generator.get_batch();

        std::cout << *batch << std::endl;

        if (batch->GetSize() == 0) {
            break;
        }
    }


}