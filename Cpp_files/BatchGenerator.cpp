#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"

#include "ChunkLoader.cpp"
#include "BatchLoader.cpp"

#include <thread>

template<typename... Args>
class BatchGenerator 
{
private:
    std::vector<std::string> cols, filters;
    size_t num_columns, chunk_size, batch_size, current_row=0, entries;

    std::string file_name, tree_name;

    std::vector<TMVA::Experimental::RTensor<float>*> x_tensors;
    size_t tensor_lengths[2] = {0,0};
    size_t training_tensor = 0, loading_tensor = 1;
    BatchLoader* batch_loader;

    std::thread loading_thread;
    bool thread_started = false, initialized = false;

    bool EoF = false;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Functions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    void LoadChunk(size_t tensor_idx) 
    {
        std::cout << "LoadChunk starting at row: " << current_row << std::endl;
        ChunkLoader<Args...> func((*x_tensors[tensor_idx]));

        // Create DataFrame        
        long long start_l = current_row;
        long long end_l = start_l + chunk_size;
        ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec(tree_name, 
                                                file_name, {start_l, std::numeric_limits<Long64_t>::max()});
        ROOT::RDataFrame x_rdf = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);

        size_t progressed_events, passed_events;

        // add filters if given
        if (filters.size() > 0) {
            auto x_filter = x_rdf.Filter(filters[0], "F1");

            for (auto i = 1; i < filters.size(); i++) {
                auto name = "F" + std::to_string(i);
                x_filter = x_filter.Filter(filters[i], name);
            }

            // add range
            auto x_ranged = x_filter.Range(chunk_size);
            auto myReport = x_ranged.Report();

            // load data
            x_ranged.Foreach(func, cols);

            // get the loading info
            progressed_events = myReport.begin()->GetAll();
            passed_events = (myReport.end()-1)->GetPass();
        }
        
        // no filters given
        else {
            
            // add range
            auto x_ranged = x_rdf.Range(chunk_size);
            auto myCount = x_ranged.Count();

            // load data
            x_ranged.Foreach(func, cols);

            // get loading info
            progressed_events = myCount.GetValue();
            passed_events = myCount.GetValue();
        }

        tensor_lengths[tensor_idx] = passed_events;
        current_row += progressed_events;
    }

    void SwapTensors() {
        size_t temp = training_tensor;
        training_tensor = loading_tensor;
        loading_tensor = temp;
    }

public:

    BatchGenerator(std::string file_name, std::string tree_name, std::vector<std::string> cols, 
                   std::vector<std::string> filters, size_t chunk_size, size_t batch_size):
        file_name(file_name), tree_name(tree_name), cols(cols), filters(filters), num_columns(cols.size()), 
        chunk_size(chunk_size), batch_size(batch_size) {
        
        // get the number of entries in the dataframe
        TFile* f = TFile::Open(file_name.c_str());
        TTree* t = f->Get<TTree>(tree_name.c_str());
        entries = t->GetEntries();

        std::cout << "found " << entries << " entries in file." << std::endl;

        x_tensors.push_back(new TMVA::Experimental::RTensor<float>({chunk_size, num_columns}));
        x_tensors.push_back(new TMVA::Experimental::RTensor<float>({chunk_size, num_columns}));

        batch_loader = new BatchLoader(batch_size, num_columns);
    }

    void init() {
        // needed to make sure nothing crashes when executing init multiple times.
        // TODO: look for better solution
        if (thread_started) {
            loading_thread.join();
        }

        EoF = false;
        training_tensor = 0;
        loading_tensor = 1;

        // loading first tensor
        LoadChunk(training_tensor);

        // loading second tensor
        loading_thread = std::thread(&BatchGenerator::LoadChunk, this, loading_tensor);
        
        thread_started = true;
        initialized = true;

        // set tensor
        batch_loader->SetTensor(x_tensors[training_tensor], tensor_lengths[training_tensor]);
    }

    void NextChunk() {
        // Join threads
        loading_thread.join();

        // Swap tensors
        SwapTensors();

        // Set T_training
        batch_loader->SetTensor(x_tensors[training_tensor], tensor_lengths[training_tensor]);
        
        // Load next Tensor if any data is left
        if (current_row < entries) {
            loading_thread = std::thread(&BatchGenerator::LoadChunk, this, loading_tensor);
        }
        else {
            EoF = true;
        }
    }
    

    // Returns the next batch of data if available. 
    // Returns empty RTensor otherwise.
    TMVA::Experimental::RTensor<float>* GetBatch()
    {   
        if (!initialized) {
            init();
        }

        // Get next batch if available
        if (batch_loader->HasData()) {
            return (*batch_loader)();
        }

        // load new chunk
        if (!EoF) {
            NextChunk();
            return GetBatch();
        }
        
        // return empty batch if all events have been used
        auto tensor = new TMVA::Experimental::RTensor<float>({0,0});
        return tensor;
    }

    bool HasData() {
        if (EoF) {
            return false;
        }

        return true;
    }
};