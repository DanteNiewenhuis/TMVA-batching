#include <iostream>
#include <vector>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

// ChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename First, typename... Rest>
class ChunkLoader
{

private:
    size_t offset = 0, vec_size_idx = 0;
    float label;
    bool add_label;
    std::vector<size_t> vec_sizes;
    
    TMVA::Experimental::RTensor<float>& x_tensor;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Value assigning
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Load the final given value into x_tensor
    // Add a label to the end of the row if given
    template <typename First_T> 
    void assign_to_tensor(First_T first)
    {
        x_tensor.GetData()[offset++] = first;

        if (add_label) {
            x_tensor.GetData()[offset++] = label;
        }
    }
    // Vector version of the previous function
    template <typename VecType> 
    void assign_to_tensor(ROOT::RVec<VecType> first)
    {
        assign_vector(first);

        if (add_label) {
            x_tensor.GetData()[offset++] = label;
        }
    }

    // Recursively loop through the given values, and load them onto the x_tensor
    template <typename First_T, typename... Rest_T> 
    void assign_to_tensor(First_T first, Rest_T... rest)
    {
        x_tensor.GetData()[offset++] = first;

        assign_to_tensor(std::forward<Rest_T>(rest)...);
    }
    // Vector version of the previous function
    template <typename VecType, typename... Rest_T> 
    void assign_to_tensor(ROOT::RVec<VecType> first, Rest_T... rest)
    {
        assign_vector(first);

        assign_to_tensor(std::forward<Rest_T>(rest)...);
    }


    // Loop through the values of a given vector and load them into the RTensor
    // Note: the given vec_size does not have to be the same size as the given vector
    //       If the size is bigger than the given vector, zeros are used as padding.
    //       If the size is smaller, the remaining values are ignored.
    template <typename VecType>
    void assign_vector(ROOT::RVec<VecType> vec){
        size_t vec_size = vec_sizes[vec_size_idx++];

        for (size_t i = 0; i < vec_size; i++) {
            x_tensor.GetData()[offset++] = vec[i];
        }
    }

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ChunkLoader(TMVA::Experimental::RTensor<float>& x_tensor, std::vector<size_t> vec_sizes = std::vector<size_t>(), 
                bool add_label=false, float label=0)
        : x_tensor(x_tensor), vec_sizes(vec_sizes), add_label(add_label), label(label)
    {}

    void operator()(First first, Rest... rest) 
    {
        vec_size_idx = 0;
        assign_to_tensor(std::forward<First>(first), std::forward<Rest>(rest)...);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void SetLabel(float l) {label = l;}
};