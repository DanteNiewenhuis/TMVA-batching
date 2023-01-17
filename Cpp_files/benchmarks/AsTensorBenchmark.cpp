#include <iostream>
#include <vector>


#include "TMVA/RTensor.hxx"
#include "TMVA/RTensorUtils.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "ROOT/RDataFrame.hxx"

#include "ChunkLoader.cpp"
#include <chrono>

void AsTensorBenchmark() {
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("test_tree", "data/Higgs_data_full.root");

    std::vector<std::string> cols = x_rdf.GetColumnNames();

    size_t chunk_size = 100000, num_columns = 29, start_row = 1900000;

    TMVA::Experimental::RTensor<float> x_tensor({chunk_size, num_columns});

    std::chrono::duration<double> elapsed_seconds;

    auto start = std::chrono::steady_clock::now();
    // ChunkLoader
    ChunkLoader<float, std::make_index_sequence<29>>
        func(x_tensor, num_columns, chunk_size, 0);

    x_rdf.Range(start_row, start_row + chunk_size).Foreach(func, cols);

    auto end = std::chrono::steady_clock::now();

    elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << x_tensor.GetSize() << std::endl;

    // start = std::chrono::steady_clock::now();
    // // AsTensor
    // x_tensor = TMVA::Experimental::AsTensor<float>(x_rdf);

    // end = std::chrono::steady_clock::now();

    // elapsed_seconds = end-start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    // std::cout << x_tensor.GetSize() << std::endl;

    ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec("test_tree", 
                                                "data/Higgs_data_full.root", {1900000, 2000000});

    ROOT::RDataFrame x_rdf_2 = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);


    start = std::chrono::steady_clock::now();

    // DatasetSpec
    x_rdf_2.Foreach(func, cols);

    end = std::chrono::steady_clock::now();

    elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    std::cout << x_tensor.GetSize() << std::endl;


}




int main() {
    return 0;
}
