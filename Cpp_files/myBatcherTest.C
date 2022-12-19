#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

// Primary template for the ChunkLoader class. 
// Required for the second class template to work
template <typename F, typename U>
class ChunkLoader;

// ChunkLoader class used to load content of a ROOT::RDataFrame onto a TMVA::Experimental::RTensor.
template <typename T, std::size_t... N>
class ChunkLoader<T, std::index_sequence<N...>>
{
    // Magic used to make make_index_sequence work.
    // Code is based on the SofieFunctorHelper
    template <std::size_t Idx>
    using AlwaysT = T;

    std::vector<std::vector<T>> fInput;

private:
    size_t num_rows, num_columns, current_row = 0;
    bool random_order;

    std::vector<size_t> row_order;
    TMVA::Experimental::RTensor<float>& x_tensor;

public:
    ChunkLoader(TMVA::Experimental::RTensor<float>& x_tensor, const size_t num_columns, const size_t num_rows, bool random_order=true)
        : x_tensor(x_tensor), num_columns(num_columns), num_rows(num_rows), random_order(random_order)
    {
        // Create a vector with elements 0...num_rows
        row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0);


        if (random_order) {
            std::random_shuffle(row_order.begin(), row_order.end());
        }
    }

    // Assign the values of a given row to the TMVA::Experimental::RTensor
    template <typename First_T>
    void assign_to_tensor(size_t offset, size_t i, First_T first)
    {
        x_tensor.GetData()[offset + i] = first;
    }

    template <typename First_T, typename... Rest_T>
    void assign_to_tensor(size_t offset, size_t i, First_T first, Rest_T... rest)
    {
        x_tensor.GetData()[offset + i] = first;
        assign_to_tensor(offset, ++i, std::forward<Rest_T>(rest)...);
    }

    // Load a given row into the TMVA::Experimental::Rtensor
    void operator()(AlwaysT<N>... args)
    {
        if (current_row >= num_rows)
            return;

        assign_to_tensor(row_order[current_row] * num_columns , 0, std::forward<AlwaysT<N>>(args)...);

        current_row++;
    }
};

class Generator_t
{
private:
    size_t current_row = 0, batch_size, num_rows, num_columns;
    TMVA::Experimental::RTensor<float>& x_tensor;
    bool drop_last;

public:

    Generator_t(TMVA::Experimental::RTensor<float>& x_tensor, const size_t batch_size, const size_t num_rows, 
                const size_t num_columns, bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_rows(num_rows), num_columns(num_columns), drop_last(drop_last) {}

    // Return a batch from the data
    TMVA::Experimental::RTensor<float> operator()()
    {
        if (current_row + batch_size <= num_rows)
        {
            unsigned long offset = current_row * num_columns;
            TMVA::Experimental::RTensor<float> x_batch(x_tensor.GetData() + offset, {batch_size, num_columns});

            current_row += batch_size;
            return x_batch;
        }
        else
        {            
            if (drop_last) {
                
                // Return empty batch
                return x_tensor.Slice({{0, 0}, {0, 0}});
            }

            unsigned long offset = current_row * num_columns;
            TMVA::Experimental::RTensor<float> x_batch(x_tensor.GetData() + offset, {(num_rows-current_row), num_columns});

            return x_batch;
        }
    }

    bool HasData() {return (current_row + batch_size <= num_rows);}
};

TMVA::Experimental::RTensor<float> load_data(
                ROOT::RDataFrame x_rdf, std::vector<std::string> cols, const size_t num_columns, 
                const size_t num_rows, const size_t start_row = 0, bool random_order=true) 
{

    TMVA::Experimental::RTensor<float> x_tensor({num_rows, num_columns});
    
    // Fill the RTensor with the data from the RDataFrame
    ChunkLoader<float, std::make_index_sequence<4>>
        func(x_tensor, num_columns, num_rows, random_order);

    // TODO: think about how to add this
    x_rdf.Range(start_row, start_row + num_rows).Foreach(func, cols);

    return x_tensor;
}


void myBatcherTest()
{
    // define variables
    std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv", "m_lv"};
    size_t batch_size = 2, start_row = 5, num_rows = 5, num_columns = cols.size();
    bool random_order = false, drop_last = false;

    // TODO remove the need to create the tensor here
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("testTree", "testFile.root", cols);

    // Make this return a tensor pointer
    TMVA::Experimental::RTensor<float> x_tensor = load_data(x_rdf, cols, num_columns, num_rows, start_row, random_order);

    std::cout << x_tensor << std::endl;

    // define generator
    Generator_t generator(x_tensor, batch_size, num_rows, num_columns, drop_last);
    
    // while (generator.HasData()) {
    //     auto batch = generator();

    //     std::cout << batch << std::endl;
    // }
}

// int main() {
//     myBatcher();
// }