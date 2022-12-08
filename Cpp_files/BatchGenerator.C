#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

class BatchGeneratorHelper
{
private:
    size_t current_row = 0, batch_size, num_rows = 0, num_columns;
    TMVA::Experimental::RTensor<float>* x_tensor;
    TMVA::Experimental::RTensor<float>* x_batch;
    bool drop_last;

    std::vector<size_t> row_order;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchGeneratorHelper(const size_t batch_size, const size_t num_columns, bool drop_last=true) 
                : batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {
                    x_batch = new TMVA::Experimental::RTensor<float>({batch_size, num_columns});
                }
    
    BatchGeneratorHelper(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_columns, 
                bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {}

    BatchGeneratorHelper(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_rows, 
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


class BatchGenerator 
{
private:
    ROOT::RDataFrame& x_rdf;
    std::vector<std::string> cols;
    size_t num_columns, chunk_size, batch_size, current_chunk=0;

    TMVA::Experimental::RTensor<float>* x_tensor;
    BatchGeneratorHelper* generator;

public:
    BatchGenerator(ROOT::RDataFrame& x_rdf, std::vector<std::string> cols, size_t chunk_size, size_t batch_size):
        x_rdf(x_rdf), cols(cols), num_columns(cols.size()), chunk_size(chunk_size), batch_size(batch_size) {
        
        x_tensor = new TMVA::Experimental::RTensor<float>({chunk_size, num_columns});
        generator = new BatchGeneratorHelper(batch_size, num_columns);
    }

    void load_chunk() 
    {
        size_t start_row = current_chunk * chunk_size;
        DataLoader<float, std::make_index_sequence<4>> func((*x_tensor), num_columns, start_row + chunk_size, start_row);

        x_rdf.Range(start_row, start_row + chunk_size).Foreach(func, cols);
        
        generator->SetTensor(x_tensor, chunk_size);

        current_chunk++;
    }

    TMVA::Experimental::RTensor<float>* get_batch()
    {
        if (generator->HasData()) {
            return (*generator)();
        }
        if (current_chunk < 10) {
            load_chunk();
            return get_batch();
        }
        
        auto tensor = new TMVA::Experimental::RTensor<float>({0,0});
        return tensor;
    }
};