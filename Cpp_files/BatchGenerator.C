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
    TMVA::Experimental::RTensor<float>* x_batch;
    bool drop_last;

    std::vector<size_t> row_order;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchGenerator(const size_t batch_size, const size_t num_columns, bool drop_last=true) 
                : batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {
                    x_batch = new TMVA::Experimental::RTensor<float>({batch_size, num_columns});
                }
    
    BatchGenerator(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_columns, 
                bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {}

    BatchGenerator(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_rows, 
                const size_t num_columns, bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_rows(num_rows), num_columns(num_columns), drop_last(drop_last) {}

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch function
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void fillBatch(std::vector<size_t> idx) {
        size_t offset;
        for (int i = 0; i < batch_size; i++) {
            offset = idx[i]*num_columns;

            // Look at std::copy

            std::copy(x_tensor->GetData() + (idx[i]*num_columns), x_tensor->GetData() + ((idx[i]+1)*num_columns), x_batch->GetData() + i*num_columns);

        }
    }

    void randomize_order() {
        std::random_shuffle(row_order.begin(), row_order.end());
    }

    size_t next() 
    {
        if (current_row >= num_rows) {
            randomize_order();
            current_row = 0;
        }
        return row_order[current_row++];
    }

    // TMVA::Experimental::RTensor<float>* getBatch() {
    //     std::vector<size_t> idx(batch_size);

    //     for (int i = 0; i < batch_size; i++) {
    //         idx[i] = next();
    //     }

    //     fillBatch(idx);

    //     return x_batch;
    // }

    TMVA::Experimental::RTensor<float>* operator()()
    {
        if (current_row + batch_size <= num_rows)
        {
            std::vector<size_t> idx(batch_size);

            for (int i = 0; i < batch_size; i++) {
                idx[i] = next();
            }

            fillBatch(idx);

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
        randomize_order();
    }

};
