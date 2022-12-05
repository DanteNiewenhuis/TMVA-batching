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
    size_t final_row, num_columns, current_row = 0;
    float label;
    bool add_label;
    

    std::vector<size_t> row_order;
    TMVA::Experimental::RTensor<float>& x_tensor;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    DataLoader(TMVA::Experimental::RTensor<float>& x_tensor, const size_t num_columns, const size_t final_row, size_t starting_row=0, bool add_label=false, float label=0)
        : x_tensor(x_tensor), num_columns(num_columns), final_row(final_row), current_row(starting_row), add_label(add_label), label(label)
    {}

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Value assigning
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Assign the values of a given row to the TMVA::Experimental::RTensor
    template <typename First_T>
    void assign_to_tensor(size_t offset, size_t i, First_T first)
    {
        x_tensor.GetData()[offset + i] = first;

        if (add_label) {
            x_tensor.GetData()[offset + i + 1] = label;
        }
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
        if (current_row >= final_row)
            return;

        assign_to_tensor(current_row * num_columns , 0, std::forward<AlwaysT<N>>(values)...);

        current_row++;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t GetCurrentRow() { return current_row;}
    void SetCurrentRow(size_t i) {current_row = i;}

    void SetLabel(float l) {label = l;}
};