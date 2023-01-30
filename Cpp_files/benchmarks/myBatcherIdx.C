#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

using namespace ROOT;
using namespace TMVA::Experimental;

// Primary template for the ChunkLoader class. 
// Required for the second class template to work
template <typename F, typename U>
class ChunkLoader;

// ChunkLoader class used to load content of a RDataFrame onto a RTensor.
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

    vector<size_t> row_order;
    TMVA::Experimental::RTensor<float> *x_tensor;

public:
    ChunkLoader(RTensor<float> *x_tensor, const size_t num_columns, const size_t num_rows, bool random_order=true)
        : x_tensor(x_tensor), num_columns(num_columns), num_rows(num_rows), random_order(random_order)
    {
        // Create a vector with elements 0...num_rows
        row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0);


        if (random_order) {
            std::random_shuffle(row_order.begin(), row_order.end());
        }
    }

    // Assign the values of a given row to the RTensor
    template <typename First_T>
    void assign_to_tensor(size_t offset, size_t i, First_T first)
    {
        x_tensor->GetData()[offset + i] = first;
    }

    template <typename First_T, typename... Rest_T>
    void assign_to_tensor(size_t offset, size_t i, First_T first, Rest_T... rest)
    {
        x_tensor->GetData()[offset + i] = first;
        assign_to_tensor(offset, ++i, std::forward<Rest_T>(rest)...);
    }

    // Load a given row into the Rtensor
    void operator()(AlwaysT<N>... args)
    {
        if (current_row >= num_rows)
            return;

        assign_to_tensor(current_row * num_columns , 0, std::forward<AlwaysT<N>>(args)...);

        current_row++;
    }
};

class Generator_t
{
private:
    size_t current_row = 0, epoch=0, batch_size, num_rows, num_columns;
    RTensor<float>* x_tensor;
    bool drop_last;

    vector<size_t> row_order;
    TMVA::Experimental::RTensor<float> *x_batch;

public:

    Generator_t(RDataFrame &x_rdf, std::vector<std::string> cols, const size_t batch_size, 
                const size_t num_rows, bool random_order = true, bool drop_last=true) 
                : batch_size(batch_size), num_rows(num_rows), num_columns(cols.size()), drop_last(drop_last)
    {
        x_tensor = new RTensor<float>({num_rows, num_columns});

        x_batch = new RTensor<float>({batch_size, num_columns});
        // Create a vector with elements 0...num_rows
        row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0);
        RandomizeOrder();

        // Fill the RTensor with the data from the RDataFrame
        ChunkLoader<float, std::make_index_sequence<4>>
            func(x_tensor, num_columns, num_rows, random_order);
        x_rdf.Foreach(func, cols);
    }

    void RandomizeOrder() {
        std::random_shuffle(row_order.begin(), row_order.end());
    }


    void FillBatch(std::vector<size_t> idx) {
        size_t offset;
        for (int i = 0; i < batch_size; i++) {
            offset = idx[i]*num_columns;

            std::copy(x_tensor->GetData() + (idx[i]*num_columns), x_tensor->GetData() + ((idx[i]+1)*num_columns), x_batch->GetData() + i*num_columns);
        }
    }

    size_t next() 
    {
        if (current_row >= num_rows) {
            std::cout << "New Epoch" << std::endl;
            RandomizeOrder();
            current_row = 0;
        }
        return row_order[current_row++];
    }

    RTensor<float>* getBatch() {
        std::vector<size_t> idx(batch_size);

        for (int i = 0; i < batch_size; i++) {
            idx[i] = next();
        }

        FillBatch(idx);

        return x_batch;
    }

    // Return a batch from the data
    // RTensor<float> operator()()
    // {
    //     if (current_row + batch_size < num_rows)
    //     {
    //         unsigned long offset = current_row * num_columns;
    //         RTensor<float> x_batch(x_tensor->GetData() + offset, {batch_size, num_columns});

    //         current_row += batch_size;
    //         return x_batch;
    //     }
    //     else
    //     {
    //         epoch++;
            
    //         if (drop_last) {
                
    //             // Return empty batch
    //             current_row = 0;
    //             return x_tensor->Slice({{0, 0}, {0, 0}});
    //         }

    //         unsigned long offset = current_row * num_columns;
    //         RTensor<float> x_batch(x_tensor->GetData() + offset, {(num_rows-current_row), num_columns});

    //         current_row = 0;
    //         return x_batch;
    //     }
    // }

    size_t GetEpoch() {return epoch;}
};

void myBatcherIdx()
{
    // define variables
    std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv", "m_lv"};
    int batch_size = 256, num_rows = 100000, epochs = 2, num_cols = cols.size();
    bool random_order = true, drop_last = false;

    // RDataFrame x_rdf = RDataFrame("testTree", "testFile.root", cols);


    RDataFrame x_rdf = RDataFrame("sig_tree", "Higgs_data_full.root", cols);

    // define generator
    Generator_t generator(x_rdf, cols, batch_size, num_rows, random_order, drop_last);

    int i = 0;
    // generate batches
    for (i = 0; i < 1000; i++)
    {
        auto batch = generator.getBatch();

        std::cout << "Batch " << i << std::endl;
        // std::cout << "Batch " << i << ": " << (*batch) << std::endl;
        i++;
    }
}

// int main() {
//     myBatcher();
// }