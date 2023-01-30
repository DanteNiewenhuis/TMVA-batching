// C++ Program to demonstrate 
// a multiprocessing environment.
#include <iostream>
#include <vector>

#include <thread>  
#include <unistd.h>


#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"

// #include "../ChunkLoader.cpp"
// #include "../BatchLoader.cpp"

#include "../BatchGenerator_threaded.cpp"
// #include "../BatchGenerator.cpp"
using namespace std;

void load_RTensor(TMVA::Experimental::RTensor<float> x_tensor, std::vector<std::string> cols, 
                  size_t chunk_size, std::string tree_name, std::string file_name) {
    size_t current_row = 0;

    // Create DatasetSpec        
    long long start_l = current_row;
    long long end_l = start_l + chunk_size;
    ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec(tree_name, 
                                            file_name, {start_l, std::numeric_limits<Long64_t>::max()});

    // Create DataFrame
    ROOT::RDataFrame x_rdf = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);
    auto x_ranged = x_rdf.Range(chunk_size);

    // Create Chunkloader and load data to x_tensor
    ChunkLoader<float&, float&, float&, float&, float&, float&, float&, 
                float&, float&, float&, float&, float&, float&, float&, 
                float&, float&, float&, float&, float&, float&, float&, 
                float&, float&, float&, float&, float&, float&, float&, float&> func(x_tensor);
    x_ranged.Foreach(func, cols);    

    std::cout << "tensor loaded 1" << std::endl;
}

void batching(BatchLoader batch_loader, size_t chunk_size) {

    TMVA::Experimental::RTensor<float>* batch;

    for (int i = 0; i < 20; i++) {

        while (batch_loader.HasData()) {
            batch = batch_loader();
        }
        batch_loader.Reset(chunk_size);
    }

}


void generator() {
    // Set parameters
    std::vector<std::string> cols = {"Type", "jet1_btag", "jet1_eta", "jet1_phi", "jet1_pt", "jet2_btag", 
                                     "jet2_eta", "jet2_phi", "jet2_pt", "jet3_btag", "jet3_eta", "jet3_phi", 
                                     "jet3_pt", "jet4_btag", "jet4_eta", "jet4_phi", "jet4_pt", "lepton_eta", 
                                     "lepton_pT", "lepton_phi", "m_bb", "m_jj", "m_jjj", "m_jlv", "m_lv", 
                                     "m_wbb", "m_wwbb", "missing_energy_magnitude", "missing_energy_phi" }; 
    size_t batch_size = 10, chunk_size = 2000000, current_row = 0, num_columns = cols.size();

    std::string file_name = "../../data/Higgs_data_full.root";
    std::string tree_name = "test_tree";

    BatchLoader batch_loader(batch_size, num_columns);

    // Create Tensor
    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});
    TMVA::Experimental::RTensor<float> x_tensor2({chunk_size, num_columns});

    load_RTensor(x_tensor, cols, chunk_size, tree_name, file_name);

    batch_loader.SetTensor(&x_tensor, chunk_size);

    std::thread t1(batching, batch_loader, chunk_size);

    std::thread t2(load_RTensor, x_tensor2, cols, chunk_size, tree_name, file_name);

    t1.join();
    t2.join();

    // batching(batch_loader, chunk_size);

    // load_RTensor(x_tensor2, cols, chunk_size, tree_name, file_name);


    // batch_loader.SetTensor(&x_tensor2, chunk_size);

}



void load_RTensor2() {
    // Set parameters
    std::vector<std::string> cols = {"fjet_C2", "fjet_D2", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3", 
                                     "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_Split12", "fjet_Split23", 
                                     "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta", 
                                     "fjet_ThrustMaj", "fjet_eta", "fjet_m", "fjet_phi", "fjet_pt", 
                                     "weights", "labels"}; 
    size_t batch_size = 10, chunk_size = 2000000, current_row = 0, num_columns = cols.size();

    auto file_name = "../../data/h5train_combined.root";
    auto tree_name = "sig_tree";

    // Create Tensor
    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});
    
    // Create DatasetSpec        
    long long start_l = current_row;
    long long end_l = start_l + chunk_size;
    ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec(tree_name, 
                                            file_name, {start_l, std::numeric_limits<Long64_t>::max()});

    // Create DataFrame
    ROOT::RDataFrame x_rdf = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);
    auto x_ranged = x_rdf.Range(chunk_size);

    // Create Chunkloader and load data to x_tensor
    ChunkLoader<float&, float&, float&, float&, float&, float&, float&, 
                float&, float&, float&, float&, float&, float&, float&, 
                float&, float&, float&, float&, float&, float&, int&> func(x_tensor);
    x_ranged.Foreach(func, cols);    

    std::cout << "tensor loaded 2" << std::endl;

}

void printA(std::string x) {
    std::cout << "printing A: " << x << std::endl;
}

void printB() {
    std::cout << "printing B" << std::endl;
}

void Test()
{
    // Define variables
    // std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv"}; 
    std::vector<std::string> cols = {"Type", "lepton_pT"}; 
    // std::vector<std::string> filters = {"m_jj < 0.9", "m_jj > 0.4"};
    std::vector<std::string> filters = {};
    size_t batch_size = 10, chunk_size = 100, max_chunks = 20000, num_columns = cols.size();


    auto file_name = "../../data/Higgs_data_full.root";
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


int main()
{
    Test();

    // std::thread t1(load_RTensor);
    // std::thread t2(load_RTensor2);

    // t1.join();
    // t2.join();

    // std::thread t1(printA, "test");
    // std::thread t2(printB);

    // t1.join();
    // t2.join();

    // load_RTensor();
    // load_RTensor2();

    // int x = 5;
    // pid_t c_pid = fork();
  
    // if (c_pid == -1) {
    //     perror("fork");
    //     exit(EXIT_FAILURE);
    // }
    // else if (c_pid > 0) {
    //     // parent 
    //     load_RTensor();
    // }
    // else {
    //     // child 
    //     load_RTensor2();
    // }
  
    // return 0;
}