#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

// Include ROOT files
#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

// Include my classes
#include "ChunkLoader.cpp"
#include "BatchGenerator.cpp"

void loadTwoRDataFrames()
{
    // Define variables
    std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv", "m_lv"};
    size_t batch_size = 10, start_row = 0, num_rows = 20, num_columns = cols.size() + 1;

    size_t file_rows = num_rows/2;

    // Load the RDataFrame and create a new tensor
    ROOT::RDataFrame x_rdf_1 = ROOT::RDataFrame("sig_tree", "data/r0-20.root", cols);
    ROOT::RDataFrame x_rdf_2 = ROOT::RDataFrame("sig_tree", "data/r10-20.root", cols);

    TMVA::Experimental::RTensor<float> x_tensor({num_rows, num_columns});

    // Fill the RTensor with the data from the RDataFrame
    ChunkLoader<float, std::make_index_sequence<4>>
        func(x_tensor, num_columns, num_rows, 0);

    x_rdf_1.Range(start_row, start_row + file_rows).Foreach(func, cols);

    // Set the starting row and label for the second dataframe
    func.SetCurrentRow(file_rows);
    func.SetLabel(1);

    x_rdf_2.Range(start_row, start_row + file_rows).Foreach(func, cols);

    std::cout << "All data" << std::endl;
    std::cout << x_tensor << std::endl << std::endl;

    // define generator
    BatchGenerator* generator = new BatchGenerator(batch_size, num_columns);

    generator->SetTensor(&x_tensor, num_rows);

    // Generate new batches until all data has been returned
    size_t i = 0;
    while (generator->HasData()) {
        auto batch = (*generator)();

        std::cout << "batch " << i << ": " << std::endl;
        std::cout << (*batch) << std::endl << std::endl;
        i++;
    }
}

int main() {
    loadTwoRDataFrames();
}