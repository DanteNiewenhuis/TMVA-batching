#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

// Primary template for the DataLoader class. 
// Required for the second class template to work
template <typename F, typename U>
class DataLoader;

// Dataloader class used to load content of a RDataFrame onto a RTensor.
template <typename T, std::size_t... N>
class DataLoader<T, std::index_sequence<N...>>
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
    DataLoader(TMVA::Experimental::RTensor<float>& x_tensor, const size_t num_columns, const size_t num_rows, bool random_order=true)
        : x_tensor(x_tensor), num_columns(num_columns), num_rows(num_rows), random_order(random_order)
    {
        // Create a vector with elements 0...num_rows
        row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0);


        // Randomize the order
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

    // Load the values of a row onto a random row of the Tensor
    void operator()(AlwaysT<N>... values)
    {
        if (current_row >= num_rows)
            return;

        assign_to_tensor(row_order[current_row] * num_columns , 0, std::forward<AlwaysT<N>>(values)...);

        current_row++;
    }

    size_t GetCurrentRow() 
    {
        return current_row;
    }
};

class Generator_t
{
private:
    size_t current_row = 0, batch_size, num_rows, num_columns;
    TMVA::Experimental::RTensor<float>* x_tensor;
    bool drop_last;

public:

    Generator_t(const size_t batch_size, const size_t num_columns, bool drop_last=true) 
                : batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {}
    
    Generator_t(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_columns, 
                bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {}

    Generator_t(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_rows, 
                const size_t num_columns, bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_rows(num_rows), num_columns(num_columns), drop_last(drop_last) {}

    // Return a batch from the data
    TMVA::Experimental::RTensor<float> operator()()
    {
        if (current_row + batch_size <= num_rows)
        {
            unsigned long offset = current_row * num_columns;
            TMVA::Experimental::RTensor<float> x_batch(x_tensor->GetData() + offset, {batch_size, num_columns});

            current_row += batch_size;
            return x_batch;
        }
        else
        {            
            // TODO: Implement drop_last
            return x_tensor->Slice({{0, 0}, {0, 0}});
        }
    }

    void SetNumRows(size_t r) {num_rows = r;}

    void Reset(size_t num_rows) {
        this->num_rows = num_rows;
        this->current_row = 0;
    }

    void SetTensor(TMVA::Experimental::RTensor<float>* x_tensor, const size_t num_rows) {
        this->x_tensor = x_tensor;
        this->num_rows = num_rows;
        this->current_row = 0;
    }

    bool HasData() {return (current_row + batch_size <= num_rows);}
};


void myBatcher()
{
    // Define variables
    std::vector<std::string> cols = {"m_jj", "m_jjj", "m_jlv", "m_lv"};
    size_t batch_size = 10, start_row = 0, num_rows = 10, num_columns = cols.size();
    bool random_order = false, drop_last = false;

    // Load the RDataFrame and create a new tensor
    ROOT::RDataFrame x_rdf = ROOT::RDataFrame("testTree", "data/testFile.root", cols);
    // ROOT::RDataFrame x_rdf = ROOT::RDataFrame("sig_tree", "Higgs_data.root", cols);
    TMVA::Experimental::RTensor<float> x_tensor({num_rows, num_columns});

    // Fill the RTensor with the data from the RDataFrame
    DataLoader<float, std::make_index_sequence<4>>
        func(x_tensor, num_columns, num_rows, random_order);

    x_rdf.Range(start_row, start_row + num_rows).Foreach(func, cols);

    // func(1, 2, 4, 5);

    // std::cout << x_tensor << std::endl;

    // std::cout << "current_row: " << func.GetCurrentRow() << std::endl;

    // define generator
    Generator_t* generator = new Generator_t(batch_size, num_columns, drop_last);

    generator->SetTensor(&x_tensor, num_rows);

    // Generate new batches until all data has been returned
    while (generator->HasData()) {
        auto batch = (*generator)();

        std::cout << "batch" << std::endl;
        std::cout << batch << std::endl;
    }
}

int main() {
    myBatcher();
}