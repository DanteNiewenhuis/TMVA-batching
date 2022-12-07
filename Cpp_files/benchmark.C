#include <iostream>
#include <fstream>
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

void load_chunkk() {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", 
        "fjet_ECF2", "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
        "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", 
        "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", 
        "fjet_pt", "weights"};

    size_t num_columns = cols.size();

    size_t chunk_size = 10, batch_size = 10;

    // Load the RDataFrame and create a new tensor
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "data/h5train_combined.root", cols);

    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});

    // Fill the RTensor with the data from the RDataFrame


    BatchGenerator* generator = new BatchGenerator(batch_size, num_columns);

    

    generator->SetTensor(&x_tensor, chunk_size);
}

void load_chunk(ROOT::RDataFrame x_rdf, TMVA::Experimental::RTensor<float>& x_tensor, BatchGenerator& generator, 
                size_t chunk_size = 1000000, size_t start_row = 0, size_t batch_size = 1000) {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", 
        "fjet_ECF2", "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
        "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", 
        "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", 
        "fjet_pt", "weights"};

    size_t num_columns = cols.size();

    DataLoader<float, std::make_index_sequence<20>>
        func(x_tensor, num_columns, chunk_size, 0);

    x_rdf.Range(start_row, start_row + chunk_size).Foreach(func, cols);

    generator.SetTensor(&x_tensor, chunk_size);
}

// void chunk_size_benchmark(){
//     std::ofstream myFile;
//     myFile.open("results/increasing_chunk_size.txt");

//     std::vector<size_t> sizes = {100, 1000, 10000, 100000, 1000000, 10000000};

//     for (size_t chunk_size : sizes) {
//         std::cout << chunk_size << std::endl;
//         auto start = std::chrono::steady_clock::now();
//         load_chunk(chunk_size);
//         auto end = std::chrono::steady_clock::now();
//         std::chrono::duration<double> elapsed_seconds = end-start;
//         std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds> (elapsed_seconds);
//         myFile << chunk_size << "," << ms.count() << std::endl;
//     }

//     myFile.close(); 
// }

// void starting_row_benchmark(){
//     std::ofstream myFile;
//     myFile.open("results/increasing_starting_row.txt");

//     size_t chunk_size = 1000000;

//     std::vector<size_t> starts;

//     for (int i = 0; i < 10; i++) {
//         starts.push_back(chunk_size*i);
//     }

//     for (size_t start_row : starts) {
//         std::cout << start_row << std::endl;
//         auto start = std::chrono::steady_clock::now();
//         load_chunk(chunk_size, start_row);
//         auto end = std::chrono::steady_clock::now();
//         std::chrono::duration<double> elapsed_seconds = end-start;
//         std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds> (elapsed_seconds);
//         myFile << chunk_size << "," << ms.count() << std::endl;
//     }

//     myFile.close(); 
// }



void benchmark()
{
    // chunk_size_benchmark();
    // starting_row_benchmark();
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", 
        "fjet_ECF2", "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
        "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", 
        "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", 
        "fjet_pt", "weights"};

    size_t num_columns = cols.size(), chunk_size = 100, batch_size = 100;




    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "data/h5train_combined.root", cols);

    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});

    // Fill the RTensor with the data from the RDataFrame

    BatchGenerator generator(batch_size, num_columns);

    load_chunk(x_rdf, x_tensor, generator, chunk_size, 0, batch_size);

    std::cout << x_tensor << std::endl;
}