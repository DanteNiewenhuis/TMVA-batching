#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <algorithm>

// Include ROOT files
#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"

// Include my classes
#include "DataLoader.C"
#include "BatchGenerator.C"

void load_chunk_base(size_t chunk_size = 1000, size_t start_row = 0, size_t batch_size = 100) {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", 
        "fjet_ECF2", "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
        "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", 
        "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", 
        "fjet_pt", "weights"};
    
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "data/h5train_combined.root");

    size_t num_columns = cols.size();
    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});

    // Fill the RTensor with the data from the RDataFrame
    DataLoader<float, std::make_index_sequence<20>>
        func(x_tensor, num_columns, chunk_size, 0);

    BatchGenerator* generator = new BatchGenerator(batch_size, num_columns);

    x_rdf.Range(start_row, start_row + chunk_size).Foreach(func, cols);

    generator->SetTensor(&x_tensor, chunk_size);
}

void load_chunk_base_2(ROOT::RDataFrame* x_rdf, TMVA::Experimental::RTensor<float>* x_tensor, 
                       size_t chunk_size = 1000, size_t start_row = 0, size_t batch_size = 100) {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", 
        "fjet_ECF2", "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
        "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", 
        "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", 
        "fjet_pt", "weights"};

    size_t num_columns = cols.size();

    // Fill the RTensor with the data from the RDataFrame
    DataLoader<float, std::make_index_sequence<20>>
        func((*x_tensor), num_columns, chunk_size, 0);

    BatchGenerator* generator = new BatchGenerator(batch_size, num_columns);

    x_rdf->Range(start_row, start_row + chunk_size).Foreach(func, cols);

    generator->SetTensor(x_tensor, chunk_size);
}

void load_chunk_spec(size_t chunk_size = 1000, size_t start_row = 0, size_t batch_size = 100) {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", 
        "fjet_ECF2", "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
        "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", 
        "fjet_Tau4_wta", "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", 
        "fjet_pt", "weights"};

    long long start = start_row;
    long long end = start + chunk_size; 
    ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec("sig_tree", 
                                                "data/h5train_combined.root", {start, end});

    ROOT::RDataFrame x_rdf = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);
    
    size_t num_columns = cols.size();
    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});

    // Fill the RTensor with the data from the RDataFrame
    DataLoader<float, std::make_index_sequence<20>>
        func(x_tensor, num_columns, chunk_size, 0);

    BatchGenerator* generator = new BatchGenerator(batch_size, num_columns);

    x_rdf.Foreach(func, cols);

    generator->SetTensor(&x_tensor, chunk_size);
}

void measure(size_t starting_row, size_t chunk_size=10000) {
    std::chrono::milliseconds ms_base;
    size_t total = 0, loops = 10;
    std::chrono::duration<double> elapsed_time;
    std::chrono::milliseconds ms;

    // // Measure base version
    // for (int i = 0; i < loops; i++) {
    //     auto start = std::chrono::steady_clock::now();
    //     load_chunk_base(chunk_size, starting_row);
    //     auto end = std::chrono::steady_clock::now();

    //     elapsed_time = end-start;
    //     ms = std::chrono::duration_cast<std::chrono::milliseconds> (elapsed_time);

    //     total += ms.count();        
    // }
    // std::cout << "Starting row: " << starting_row << " base: " << total / loops << std::endl;


    // // Measure upgr version
    // ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "data/h5train_combined.root");
    // TMVA::Experimental::RTensor<float> x_tensor({chunk_size, 20});
    // for (int i = 0; i < loops; i++) {
    //     auto start = std::chrono::steady_clock::now();
    //     load_chunk_base_2(&x_rdf, &x_tensor, chunk_size, starting_row);
    //     auto end = std::chrono::steady_clock::now();

    //     elapsed_time = end-start;
    //     ms = std::chrono::duration_cast<std::chrono::milliseconds> (elapsed_time);

    //     total += ms.count();        
    // }
    // std::cout << "Starting row: " << starting_row << " upgr: " << total / loops << std::endl;


    // Measure spec version
    total = 0;
    for (int i = 0; i < loops; i++) {
        auto start = std::chrono::steady_clock::now();
        load_chunk_spec(chunk_size, starting_row);
        auto end = std::chrono::steady_clock::now();

        elapsed_time = end-start;
        ms = std::chrono::duration_cast<std::chrono::milliseconds> (elapsed_time);
        std::cout << ms.count() << std::endl;
        total += ms.count();        
    }

    std::cout << "Starting row: " << starting_row << " spec: " << total / loops << std::endl;

}


void SpecTester() {
    size_t starting_row = 0, chunk_size = 100000;

    measure(starting_row, chunk_size);

}


int main() {
    return 0;
}