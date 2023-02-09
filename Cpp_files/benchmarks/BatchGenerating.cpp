#include <iostream>
#include <fstream>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"

// Include my classes
#include "../ChunkLoader.cpp"
#include "../BatchLoader_threaded.cpp"

// Timing
#include <chrono>

int main() {
    std::cout << "MAIN" << std::endl;
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", 
                                "fjet_ECF3", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", 
                                "fjet_Split23", "fjet_Tau1_wta", "fjet_Tau2_wta", 
                                "fjet_Tau3_wta", "fjet_Tau4_wta", "fjet_ThrustMaj", 
                                "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", "weights"};

    std::string file_name = "../../data/h5train_combined.root";
    std::string tree_name = "sig_tree";

    size_t chunk_size = 200000, batch_size = 1000, num_columns = cols.size();

    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});
    ROOT::RDataFrame x_rdf(tree_name, file_name, cols);


    size_t num_threads = 2, num_batches = 10;
    BatchLoader batch_loader(batch_size, num_columns, num_threads, num_batches);

    std::cout << num_columns << std::endl;

    ChunkLoader <float&, float&, float&, float&, float&, float&, float&, float&, float&, 
                 float&, float&, float&, float&, float&, float&, float&, float&, float&, 
                 float&, float&> func(x_tensor);

    x_rdf.Range(chunk_size).Foreach(func, cols);

    std::cout << x_tensor.GetSize() << std::endl;

    batch_loader.SetTensor(&x_tensor, chunk_size);
    batch_loader.Done();

    std::ofstream myFile;
    std::string s = "batching2.txt";
    myFile.open(s);

    myFile << "0" << std::endl;

    auto start = std::chrono::steady_clock::now();
    size_t i = 0;
    while (batch_loader.HasData()) {
        std::cout << "MainLoop => batch: " << ++i << std::endl;
        TMVA::Experimental::RTensor<float>* batch = batch_loader.GetBatch();
        // std::cout << "MainLoop => Batch: " << batch->GetSize() << std::endl;
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        myFile << elapsed_seconds.count() << std::endl;

        batch_loader.AddBatch(batch);
    }

    myFile.close();
}