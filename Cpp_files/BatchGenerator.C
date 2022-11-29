#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

class BatchGenerator
{
private:
    size_t current_row = 0, batch_size, num_rows, num_columns;
    TMVA::Experimental::RTensor<float>* x_tensor;
    bool drop_last;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchGenerator(const size_t batch_size, const size_t num_columns, bool drop_last=true) 
                : batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {}
    
    BatchGenerator(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_columns, 
                bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {}

    BatchGenerator(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_rows, 
                const size_t num_columns, bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_rows(num_rows), num_columns(num_columns), drop_last(drop_last) {}

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch function
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
    bool HasData() {return (current_row + batch_size <= num_rows);}

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

};
