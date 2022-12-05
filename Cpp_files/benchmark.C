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

// Timing
#include <chrono>

void load_data() {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", 
        "fjet_ECF2", "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
        "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", 
        "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", 
        "fjet_pt", "weights"};

    size_t batch_size = 1000, start_row = 0, chunk_size = 1000000, num_columns = cols.size();

    // Load the RDataFrame and create a new tensor
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "data/h5train_combined.root", cols);

    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});

    // Fill the RTensor with the data from the RDataFrame
    DataLoader<float, std::make_index_sequence<20>>
        func(x_tensor, num_columns, chunk_size, 0);

    BatchGenerator* generator = new BatchGenerator(batch_size, num_columns);

    x_rdf.Range(start_row, start_row + chunk_size).Foreach(func, cols);

    generator->SetTensor(&x_tensor, chunk_size);

    // Generate new batches until all data has been returned
    while (generator->HasData()) {
        auto batch = (*generator)();

        std::cout << "batch" << std::endl;
        // std::cout << (*batch) << std::endl << std::endl;
    }

    start_row = chunk_size;

    x_rdf.Range(start_row, start_row + chunk_size).Foreach(func, cols);

    generator->SetTensor(&x_tensor, chunk_size);

    // Generate new batches until all data has been returned
    while (generator->HasData()) {
        auto batch = (*generator)();

        std::cout << "batch" << std::endl;
        // std::cout << (*batch) << std::endl << std::endl;
    }
}

void benchmark()
{
    auto start = std::chrono::steady_clock::now();
    load_data();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds> (elapsed_seconds);
    std::cout << "elapsed time: " << ms.count() << "s\n";
}