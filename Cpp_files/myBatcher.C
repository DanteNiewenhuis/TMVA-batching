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

//
size_t load_dataC(TMVA::Experimental::RTensor<float>& x_tensor, ROOT::RDF::RNode x_rdf,
                std::vector<std::string> cols, const size_t num_columns, 
                const size_t chunk_rows, const size_t start_row = 0, bool random_order=true) 
{
    // Fill the RTensor with the data from the RDataFrame
    DataLoader<float, std::make_index_sequence<4>> func(x_tensor, 
                        num_columns, chunk_rows, random_order);

    auto myCount = x_rdf.Range(start_row, start_row + chunk_rows).Count();

    x_rdf.Range(start_row, start_row + chunk_rows).Foreach(func, cols);

    return myCount.GetValue();
}

void myBatcher()
{
    // Define variables
    std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv", "m_lv"};
    size_t batch_size = 2, start_row = 0, num_rows = 5, num_columns = cols.size();
    bool random_order = false, drop_last = false;

    // Load the RDataFrame and create a new tensor
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("testTree", "data/testFile.root", cols);
    auto x_filter = x_rdf.Filter("m_jj < 1");

    // ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "Higgs_data.root", cols);
    TMVA::Experimental::RTensor<float> x_tensor({num_rows, num_columns});

    // Fill the RTensor with the data from the RDataFrame
    size_t count = load_dataC(x_tensor, x_filter, cols, num_columns, num_rows, 0, false);

    // define generator
    BatchGenerator* generator = new BatchGenerator(batch_size, num_columns, drop_last);

    generator->SetTensor(&x_tensor, num_rows);

    // // Generate new batches until all data has been returned
    while (generator->HasData()) {
        auto batch = (*generator)();

        std::cout << "batch" << std::endl;
        std::cout << batch << std::endl;
    }
}

int main() {
    myBatcher();
}