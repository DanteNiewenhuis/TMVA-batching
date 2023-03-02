#include <iostream>
#include <fstream>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"

// Include my classes
#include "../ChunkLoader.cpp"
#include "../BatchLoader.cpp"

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
    TMVA::Experimental::RTensor<float> x_tensor_2({chunk_size, num_columns});
    ROOT::RDataFrame x_rdf(tree_name, file_name, cols);


    size_t num_threads = 2, num_batches = 300;
    BatchLoader batch_loader(batch_size, num_columns, num_threads, num_batches, 0.7);

    std::cout << num_columns << std::endl;

    ChunkLoader <float&, float&, float&, float&, float&, float&, float&, float&, float&, 
                 float&, float&, float&, float&, float&, float&, float&, float&, float&, 
                 float&, float&> func(x_tensor);
    ChunkLoader <float&, float&, float&, float&, float&, float&, float&, float&, float&, 
                float&, float&, float&, float&, float&, float&, float&, float&, float&, 
                float&, float&> func_2(x_tensor);
    
    x_rdf.Range(chunk_size).Foreach(func, cols);
    x_rdf.Range(chunk_size, 2*chunk_size).Foreach(func_2, cols);

    batch_loader.AddTasks(&x_tensor, chunk_size);
    batch_loader.AddTasks(&x_tensor_2, chunk_size);
    batch_loader.Done();
    batch_loader.wait_for_tasks();

    std::ofstream myFile;
    size_t delay = 0;
    // std::string s = "batching" + std::to_string(delay) + ".txt";    
    std::string s = "batching_first.txt";
    myFile.open(s);

    myFile << "0" << std::endl;

    auto start= std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    size_t i = 0;
    while (batch_loader.HasTrainData()) {
        TMVA::Experimental::RTensor<float>* batch = batch_loader.GetTrainBatch();
        std::cout << "MainLoop => train batch: " << ++i << " " << batch->GetSize() << std::endl;
        
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        myFile << elapsed_seconds.count() << std::endl;

        usleep(delay);

        delete batch;
    }

    i = 0;
    while (batch_loader.HasValidationData()) {
        TMVA::Experimental::RTensor<float>* batch = batch_loader.GetValidationBatch();
        std::cout << "MainLoop => test batch: " << ++i << " " << batch->GetSize() << std::endl;
        
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        myFile << elapsed_seconds.count() << std::endl;

        usleep(delay);

        delete batch;
    }

    std::cout << "ENDDD" << std::endl;

    myFile.close();
}