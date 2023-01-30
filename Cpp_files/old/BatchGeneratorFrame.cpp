#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TFile.h"

#include "ChunkLoader.cpp"

class BatchLoader
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
    BatchLoader(const size_t batch_size, const size_t num_columns, bool drop_last=true) 
                : batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {
                    x_batch = new TMVA::Experimental::RTensor<float>({batch_size, num_columns});
                }
    
    BatchLoader(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_columns, 
                bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_columns(num_columns), drop_last(drop_last) {}

    BatchLoader(TMVA::Experimental::RTensor<float>* x_tensor, const size_t batch_size, const size_t num_rows, 
                const size_t num_columns, bool drop_last=true) 
                : x_tensor(x_tensor), batch_size(batch_size), num_rows(num_rows), num_columns(num_columns), drop_last(drop_last) {}

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch function
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void FillBatch(std::vector<size_t> idx) {
        size_t offset;
        for (int i = 0; i < batch_size; i++) {
            offset = idx[i]*num_columns;

            // Look at std::copy

            std::copy(x_tensor->GetData() + (idx[i]*num_columns), x_tensor->GetData() + ((idx[i]+1)*num_columns), x_batch->GetData() + i*num_columns);

        }
    }

    void RandomizeOrder() {
        std::random_shuffle(row_order.begin(), row_order.end());
    }

    size_t next() 
    {
        if (current_row >= num_rows) {
            RandomizeOrder();
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


class BatchGenerator 
{
private:
    ROOT::RDataFrame* x_rdf;
    std::vector<std::string> cols;
    size_t num_columns, chunk_size, current_row=0, max_chunks, batch_size, entries;

    string file_name, tree_name;

    bool EoF = false;

    TMVA::Experimental::RTensor<float>* x_tensor;
    BatchLoader* helper;

public:
    // BatchGenerator(ROOT::RDataFrame& x_rdf, std::vector<std::string> cols, size_t chunk_size, size_t batch_size, size_t max_chunks):
    //     x_rdf(x_rdf), cols(cols), num_columns(cols.size()), chunk_size(chunk_size), max_chunks(max_chunks), batch_size(batch_size) {
        
    //     x_tensor = new TMVA::Experimental::RTensor<float>({chunk_size, num_columns});
    //     helper = new BatchLoader(batch_size, num_columns);
    // }

    BatchGenerator(string file_name, string tree_name, std::vector<std::string> cols, size_t chunk_size, size_t batch_size, size_t max_chunks):
        file_name(file_name), tree_name(tree_name), cols(cols), num_columns(cols.size()), chunk_size(chunk_size), max_chunks(max_chunks), batch_size(batch_size) {
        
        // get the number of entries in the dataframe
        TFile* f = TFile::Open(file_name.c_str());
        TTree* t = f->Get<TTree>(tree_name.c_str());
        entries = t->GetEntries();

        std::cout << entries << std::endl;

        x_rdf = new ROOT::RDataFrame(tree_name, file_name);

        x_tensor = new TMVA::Experimental::RTensor<float>({chunk_size, num_columns});
        helper = new BatchLoader(batch_size, num_columns);
    }

    void load_chunk() 
    {
        std::cout << "load_chunk " << current_row << std::endl;
        ChunkLoader<float, std::make_index_sequence<20>> func((*x_tensor), num_columns, chunk_size);

        auto myCount = x_rdf->Range(current_row, current_row + chunk_size).Count();

        x_rdf->Range(current_row, current_row + chunk_size).Foreach(func, cols);

        size_t loaded_size = myCount.GetValue();

        helper->SetTensor(x_tensor, loaded_size);

        current_row += chunk_size;
    }

    TMVA::Experimental::RTensor<float>* get_batch()
    {
        if (helper->HasData()) {
            return (*helper)();
        }

        if (current_row < entries) {
            load_chunk();
            return get_batch();
        }
        
        auto tensor = new TMVA::Experimental::RTensor<float>({0,0});
        return tensor;
    }

    bool hasData() {
        if (current_row < entries) {
            return true;
        }

        return false;
    }
};