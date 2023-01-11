#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <algorithm>

// Include ROOT files
#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

// Include my classes
// #include "ChunkLoader.cpp"
#include "BatchGeneratorSpec.cpp"
#include "ROOT/RDF/RDatasetSpec.hxx"

// Timing
#include <chrono>

void datasetspec_test(size_t chunk_size, string name) {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                    "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                    "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                    "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                    "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"};

    size_t batch_size = 2000, max_chunks = 20000;

    string file_name;
    if (name == "h5") {
        file_name = "data/h5train_combined.root";
    }
    if (name == "Higgs") {
        file_name = "data/Higgs_data_full.root";
    }
    string tree_name = "sig_tree";

    TFile* f = TFile::Open(file_name.c_str());
    TTree* t = f->Get<TTree>(tree_name.c_str());
    size_t entries = t->GetEntries();

    std::cout << entries << std::endl;

    ROOT::RDataFrame x_rdf = ROOT::RDataFrame(tree_name, file_name);
    // std::vector<std::string> cols = x_rdf.GetColumnNames();
    size_t num_columns = cols.size();

    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});
    ChunkLoader<float, std::make_index_sequence<20>> func(x_tensor, num_columns, chunk_size);

    ofstream myFile;
    string s = "results/DataFrame_DatasetSpec/DatasetSpec_" + name + "_" + std::to_string(chunk_size) + ".txt";
    myFile.open(s);
    
    for (size_t current_row = 0; current_row + chunk_size <= entries; current_row += chunk_size) {
        auto start = std::chrono::steady_clock::now();

        long long start_l = current_row;
        long long end_l = start_l + chunk_size;
        ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec(tree_name, 
                                                file_name, {start_l, std::numeric_limits<Long64_t>::max()});
        ROOT::RDataFrame x_rdf = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);

        x_rdf.Range(0, chunk_size).Foreach(func, cols);
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        std::cout << elapsed_seconds.count() << std::endl;
        myFile << elapsed_seconds.count() << std::endl;
    }

    myFile.close();
}


void dataframe_test(size_t chunk_size, string name) {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                    "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                    "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                    "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                    "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"};

    size_t batch_size = 2000, max_chunks = 20000;

    string file_name;
    if (name == "h5") {
        file_name = "data/h5train_combined.root";
    }
    if (name == "Higgs") {
        file_name = "data/Higgs_data_full.root";
    }

    string tree_name = "sig_tree";

    TFile* f = TFile::Open(file_name.c_str());
    TTree* t = f->Get<TTree>(tree_name.c_str());
    size_t entries = t->GetEntries();

    std::cout << entries << std::endl;

    ROOT::RDataFrame x_rdf = ROOT::RDataFrame(tree_name, file_name);
    // std::vector<std::string> cols = x_rdf.GetColumnNames();
    size_t num_columns = cols.size();

    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});
    ChunkLoader<float, std::make_index_sequence<20>> func(x_tensor, num_columns, chunk_size);

    string s = "results/DataFrame_DatasetSpec/DataFrame_" + name + "_" + std::to_string(chunk_size) + ".txt";

    ofstream myFile;
    myFile.open(s);
    
    for (size_t current_row = 0; current_row + chunk_size <= entries; current_row += chunk_size) {
        auto start = std::chrono::steady_clock::now();
        // ROOT::RDataFrame x_rdf = ROOT::RDataFrame(tree_name, file_name);

        x_rdf.Range(current_row, current_row + chunk_size).Foreach(func, cols);
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        std::cout << elapsed_seconds.count() << std::endl;
        myFile << elapsed_seconds.count() << std::endl;
    }

    myFile.close();
}

void generator_test(size_t chunk_size, string name) {
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                    "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                    "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                    "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                    "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"};

    size_t batch_size = 2000, max_chunks = 20000;

    string file_name;
    if (name == "h5") {
        file_name = "data/h5train_combined.root";
    }
    if (name == "Higgs") {
        file_name = "data/Higgs_data_full.root";
    }

    string tree_name = "sig_tree";

    // ROOT::RDataFrame x_rdf = ROOT::RDataFrame(tree_name, file_name);
    // std::vector<std::string> cols = x_rdf.GetColumnNames();
    size_t num_columns = cols.size();

    string s = "results/Cpp_Python/" + std::to_string(chunk_size) + "_Cpp_Spec.txt";

    ofstream myFile;
    myFile.open(s, std::ofstream::trunc);
    myFile << "0" << std::endl;

    BatchGenerator generator = BatchGenerator(file_name, tree_name, cols, chunk_size, batch_size, max_chunks);
    
    auto start = std::chrono::steady_clock::now();
    while(generator.hasData()) {
        auto batch = generator.get_batch();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        myFile << elapsed_seconds.count() << std::endl;
    }

    myFile.close();
}

void chunk_test(size_t chunk_size, string name) {
    // std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
    //                                 "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
    //                                 "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
    //                                 "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
    //                                 "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"};

    size_t batch_size = 2000, max_chunks = 20000;

    string file_name;
    if (name == "h5") {
        file_name = "data/h5train_combined.root";
    }
    if (name == "Higgs") {
        file_name = "data/Higgs_data_full.root";
    }

    string tree_name = "sig_tree";

    ROOT::RDataFrame x_rdf = ROOT::RDataFrame(tree_name, file_name);
    std::vector<std::string> cols = x_rdf.GetColumnNames();
    size_t num_columns = cols.size();

    string s = "results/chunk_test/Cpp_Frame.txt";

    ofstream myFile;
    myFile.open(s, std::ofstream::trunc);
    myFile << "0" << std::endl;

    BatchGenerator generator = BatchGenerator(file_name, tree_name, cols, chunk_size, batch_size, max_chunks);
    
    auto start = std::chrono::steady_clock::now();
    while(generator.hasData()) {
        generator.load_chunk();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        myFile << elapsed_seconds.count() << std::endl;
    }

    myFile.close();
}


void benchmark()
{
    generator_test(100000, "h5");
}