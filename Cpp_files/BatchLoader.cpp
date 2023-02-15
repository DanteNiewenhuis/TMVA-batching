#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"

class BatchLoader
{
private:
    size_t current_row = 0, num_rows = 0;
    TMVA::Experimental::RTensor<float>* x_tensor; 
    TMVA::Experimental::RTensor<float>* x_batch;

    const size_t batch_size, num_columns;

    const bool drop_last;

    std::vector<size_t> row_order;
    
    // Randomize the order of the indices
    void RandomizeOrder() {
        std::random_shuffle(row_order.begin(), row_order.end());
    }

    // Fil the batch with rows from the chunk based on the given idx
    void FillBatch(std::vector<size_t> idx) {
        size_t offset;
        for (int i = 0; i < batch_size; i++) {
            std::copy(x_tensor->GetData() + (idx[i]*num_columns), 
                      x_tensor->GetData() + ((idx[i]+1)*num_columns), 
                      x_batch->GetData() + i*num_columns);

        }
    }

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchLoader(const size_t batch_size, const size_t num_columns, const bool drop_last=true) 
                : batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {
                    x_batch = new TMVA::Experimental::RTensor<float>({batch_size, num_columns});
                }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch functions
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // return a batch of data
    TMVA::Experimental::RTensor<float>* operator()()
    {
        // current_row += batch_size;
        // return x_batch;

        if (HasData())
        {
            // Take the batch_size indices from the row_order
            std::vector<size_t> idx{row_order.begin() + current_row, row_order.begin() + current_row + batch_size};
            current_row += batch_size;

            // Fill the batchrows with the rows from the chunk
            FillBatch(idx);

            return x_batch;
        }
        else
        {            
            // TODO: Implement drop_last
            return x_batch;
        }
    }
    bool HasData() {
        return (current_row + batch_size <= num_rows);}

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

        row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0);
        RandomizeOrder();
    }

};