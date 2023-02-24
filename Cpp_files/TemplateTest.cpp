#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

// Include ROOT files
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TMVA/RTensor.hxx"

// Include my classes
#include <unistd.h>

#include <fstream>

#include "ChunkLoader.cpp"

#include <thread>

void LoadChunk(){
    std::string name = "Higgs";

    size_t batch_size = 1024, chunk_size = 5;

    double validation_split = 1;

    std::string file_name = "../data/vectorData.root";
    std::string tree_name = "sig_tree";

    ROOT::RDataFrame x_rdf = ROOT::RDataFrame(tree_name, file_name);
    std::vector<std::string> cols = x_rdf.GetColumnNames();
    std::vector<size_t> vec_sizes = {5};

    size_t num_columns = cols.size();
    for (size_t s: vec_sizes) {
        num_columns += s - 1;
    }
    
    std::cout << "num_columns: " << num_columns << std::endl; 


    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});
    ChunkLoader<int&, ROOT::RVec<int>, int&> func(x_tensor, vec_sizes);
    auto x_ranged = x_rdf.Range(chunk_size);

    // std::cout << "looping" << std::endl;
    x_ranged.Foreach(func, cols);
    // std::cout << "done" << std::endl;

    std::cout << x_tensor << std::endl;
}

void generator_test(std::string name)
{
    // std::vector<std::string> cols = {
    //     "fjet_C2",       "fjet_D2",       "fjet_ECF1",      "fjet_ECF2",
    //     "fjet_ECF3",     "fjet_L2",       "fjet_L3",        "fjet_Qw",
    //     "fjet_Split12",  "fjet_Split23",  "fjet_Tau1_wta",  "fjet_Tau2_wta",
    //     "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta",
    //     "fjet_m",        "fjet_phi",      "fjet_pt",        "weights"};

    std::thread loading_thread = std::thread(LoadChunk);
    loading_thread.join();
}

int main()
{
    generator_test("Higgs");

    return 0;
}