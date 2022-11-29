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
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Value assigning
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t GetCurrentRow() 
    {
        return current_row;
    }
};