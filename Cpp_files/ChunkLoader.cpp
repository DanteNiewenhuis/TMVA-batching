#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

// ChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename First, typename... Rest>
class ChunkLoader
{

private:
    size_t current_row = 0, offset = 0;
    float label;
    bool add_label;
    
    TMVA::Experimental::RTensor<float>& x_tensor;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ChunkLoader(TMVA::Experimental::RTensor<float>& x_tensor, bool add_label=false, float label=0)
        : x_tensor(x_tensor), add_label(add_label), label(label)
    {}

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Value assigning
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Assign the values of a given row to the TMVA::Experimental::RTensor
    template <typename First_T> 
    void assign_to_tensor(First_T first)
    {
        x_tensor.GetData()[offset++] = first;

        if (add_label) {
            x_tensor.GetData()[offset++] = label;
        }
    }

    // Load the values of a row onto a random row of the Tensor
    template <typename First_T, typename... Rest_T> 
    void assign_to_tensor(First_T first, Rest_T... rest)
    {
        x_tensor.GetData()[offset++] = first;

        assign_to_tensor(std::forward<Rest_T>(rest)...);
    }

    void operator()(First first, Rest... rest) 
    {
        assign_to_tensor(std::forward<First>(first), std::forward<Rest>(rest)...);
        current_row++;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t GetCurrentRow() { return current_row;}
    void SetCurrentRow(size_t i) {current_row = i;}

    void SetLabel(float l) {label = l;}
};