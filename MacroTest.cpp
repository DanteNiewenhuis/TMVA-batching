#include <iostream>
#include <vector>

#include <TMVA/ChunkLoader.hxx>
#include <TMVA/BatchLoader.hxx>

void MacroTest() {
    std::cout << "TeST" << std::endl;

    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("test_tree", "data/Higgs_data_full.root");


    
    size_t chunk_size = 1000;
    size_t batch_size = 2;
    std::vector<std::string> cols = x_rdf.GetColumnNames();
    size_t num_columns = cols.size();

    TMVA::Experimental::BatchLoader batch_loader = TMVA::Experimental::BatchLoader(batch_size, num_columns);

    TMVA::Experimental::RTensor<float>* x_tensor = new TMVA::Experimental::RTensor<float>({chunk_size, num_columns});

    TMVA::Experimental::ChunkLoader<float&, float&, float&, float&, float&, float&, float&, float&, float&, float&, 
                                    float&, float&, float&, float&, float&, float&, float&, float&, float&, float&, 
                                    float&, float&, float&, float&, float&, float&, float&, float&, float&> func((*x_tensor));

    x_rdf.Range(chunk_size).Foreach(func, cols);

    // std::cout << *x_tensor << std::endl;

    std::vector<size_t> row_order = std::vector<size_t>(chunk_size);
    std::iota(row_order.begin(), row_order.end(), 0);

    std::cout << row_order.size() << std::endl;

    batch_loader.CreateTrainingBatches(x_tensor, row_order);

    TMVA::Experimental::RTensor<float>* batch = batch_loader.GetTrainBatch();

    std::cout << *batch << std::endl;
}