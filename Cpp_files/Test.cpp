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
    // std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv"}; 
    std::vector<std::string> cols = {"Type", "lepton_pT"}; 
    // std::vector<std::string> filters = {"m_jj < 0.9", "m_jj > 0.4"};
    std::vector<std::string> filters = {};
    size_t batch_size = 10, chunk_size = 100, max_chunks = 20000, num_columns = cols.size();


    auto file_name = "data/Higgs_data_full.root";
    // auto file_name = "data/test.root";
    auto tree_name = "test_tree";



    // BatchGenerator<float&, int&, float&> generator(file_name, tree_name, cols, filters, chunk_size, batch_size, max_chunks);

    BatchGenerator<float&, float&> generator(file_name, tree_name, cols, filters, chunk_size, batch_size, max_chunks);

    auto batch = generator.get_batch();

    std::cout << (*batch) << std::endl;

    // ROOT::RDataFrame x_rdf(tree_name, file_name);
    // TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});
    
    // ChunkLoader<float&, int&, float&> func(x_tensor);
    
    // x_rdf.Foreach(func, cols);

    // std::cout << x_tensor << std::endl;
}