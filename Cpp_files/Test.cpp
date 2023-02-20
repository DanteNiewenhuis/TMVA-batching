#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

// Include ROOT files
#include "ROOT/RDataFrame.hxx"
#include "TMVA/RTensor.hxx"

// Include my classes
#include <unistd.h>

#include <fstream>

#include "BatchGenerator.cpp"

void generator_test(std::string name)
{
    // std::vector<std::string> cols = {
    //     "fjet_C2",       "fjet_D2",       "fjet_ECF1",      "fjet_ECF2",
    //     "fjet_ECF3",     "fjet_L2",       "fjet_L3",        "fjet_Qw",
    //     "fjet_Split12",  "fjet_Split23",  "fjet_Tau1_wta",  "fjet_Tau2_wta",
    //     "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta",
    //     "fjet_m",        "fjet_phi",      "fjet_pt",        "weights"};

    size_t batch_size = 1024, chunk_size = 200000;

    double train_ratio = 1;

    std::string file_name;
    if (name == "h5")
    {
        file_name = "../data/h5train_combined.root";
    }
    if (name == "Higgs")
    {
        file_name = "../data/Higgs_data_full.root";
    }

    std::string tree_name = "sig_tree";

    ROOT::RDataFrame x_rdf = ROOT::RDataFrame(tree_name, file_name);
    std::vector<std::string> cols = x_rdf.GetColumnNames();
    size_t num_columns = cols.size();

    BatchGenerator generator =
        BatchGenerator<float&, float&, float&, float&, float&, float&, float&,
                       float&, float&, float&, float&, float&, float&, float&,
                       float&, float&, float&, float&, float&, float&, float&, 
                       float&, float&, float&, float&, float&, float&, float&>(
            file_name, tree_name, cols, {}, chunk_size, batch_size, train_ratio);

    generator.init();

    std::cout << "MAIN => Chunks loaded" << std::endl;
    
    // usleep(1000000);

    std::cout << "slept" << std::endl;

    std::ofstream myFile;
    size_t delay = 0;
    // std::string s = "batching" + std::to_string(delay) + ".txt";    
    std::string s = "batching_test.txt";
    myFile.open(s);

    myFile << "0" << std::endl;

    auto start= std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    size_t i = 0;
    while(generator.HasTrainData()) {
        auto batch = generator.GetTrainBatch();

        std::cout << "train batch " << ++i << " " << batch->GetSize() << std::endl;

        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        myFile << elapsed_seconds.count() << std::endl;

        delete batch;
        usleep(10000);
    }

    myFile.close();

    i = 0;
    while(generator.HasValidationData()) {
        auto batch = generator.GetValidationBatch();

        std::cout << "test batch " << ++i << " " << batch->GetSize() << std::endl;
    }
}

int main()
{
    generator_test("Higgs");

    return 0;
}