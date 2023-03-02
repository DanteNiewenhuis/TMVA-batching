#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
// #include <TROOT.h>

#include "ChunkLoader.cpp"
#include "BatchLoader.cpp"

#include <thread>

#include <chrono>
#include <unistd.h>

// ROOT::EnableThreadSafety();


template<typename... Args>
class BatchGenerator 
{
private:
    std::vector<std::string> cols, filters;
    size_t num_columns, chunk_size, max_chunks, batch_size, current_row=0, entries;

    std::string file_name, tree_name;
    
    BatchLoader* batch_loader;

    std::thread* loading_thread = 0;
    bool initialized = false;

    bool EoF = false, use_whole_file = true;
    double validation_split;

    TMVA::Experimental::RTensor<float>* previous_batch = 0;
    TMVA::Experimental::RTensor<float>* x_tensor;

    std::vector<size_t> vec_sizes;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Functions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Load chunk_size rows of the given RDataFrame into a RTensor.
    // After, the chunk of data is split into batches of data.
    void LoadChunk() 
    {
        
        ChunkLoader<Args...> func((*x_tensor), vec_sizes);

        // Create DataFrame
        long long start_l = current_row;
        long long end_l = start_l + chunk_size;
        ROOT::RDF::Experimental::RDatasetSpec x_spec = ROOT::RDF::Experimental::RDatasetSpec().AddGroup({"",tree_name,
                                                file_name}).WithGlobalRange( {start_l, std::numeric_limits<Long64_t>::max()});
        ROOT::RDataFrame x_rdf(x_spec);

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
        
        // std::cout << "BatchGenerator::init => tensor: " << x_tensor << std::endl;

        current_row += progressed_events;

        batch_loader->CreateBatches(x_tensor, passed_events);
    }

public:

    BatchGenerator(std::string file_name, std::string tree_name, std::vector<std::string> cols, 
                   std::vector<std::string> filters, size_t chunk_size, size_t batch_size, std::vector<size_t> vec_sizes = {}, double validation_split=0.0, 
                   size_t max_chunks = 0, size_t num_columns = 0):
        file_name(file_name), tree_name(tree_name), cols(cols), filters(filters), num_columns(num_columns), 
        chunk_size(chunk_size), batch_size(batch_size), vec_sizes(vec_sizes), validation_split(validation_split), max_chunks(max_chunks) {
        
        if (max_chunks > 0) {use_whole_file = false;};

        if (num_columns == 0){
            num_columns = cols.size();
        }

        std::cout << "BatchGenerator => num_columns: " << num_columns << std::endl;
        std::cout << "BatchGenerator => validation_split: " << validation_split << std::endl;

        // get the number of entries in the dataframe
        TFile* f = TFile::Open(file_name.c_str());
        TTree* t = f->Get<TTree>(tree_name.c_str());
        entries = t->GetEntries();

        std::cout << "BatchGenerator => found " << entries << " entries in file." << std::endl;

        batch_loader = new BatchLoader(batch_size, num_columns, validation_split);

        x_tensor = new TMVA::Experimental::RTensor<float>({chunk_size, num_columns});
    }

    ~BatchGenerator () {
        StopLoading();
    } 

    void StopLoading() {
        if (loading_thread != 0) {
            loading_thread->join();
            delete loading_thread;
            loading_thread = 0;
        }
    }

    void init() {
        StopLoading(); // make sure the thread is currently not loading
        
        current_row = 0;
        batch_loader->Activate();
        loading_thread = new std::thread(&BatchGenerator::LoadChunks, this);
    }

    // Returns the next batch of data if available. 
    // Returns empty RTensor otherwise.
    TMVA::Experimental::RTensor<float>* GetTrainBatch()
    {   
        if (previous_batch != 0) {
            delete previous_batch;
            previous_batch = 0;
        }

        // Get next batch if available
        if (batch_loader->HasTrainData()) {
            TMVA::Experimental::RTensor<float>* batch = batch_loader->GetTrainBatch();
            previous_batch = batch;
            return batch;
        }

        // return empty batch if all events have been used
        return new TMVA::Experimental::RTensor<float>({0,0});
    }

    // Returns the next batch of data if available. 
    // Returns empty RTensor otherwise.
    TMVA::Experimental::RTensor<float>* GetValidationBatch()
    {   
        if (previous_batch != 0) {
            delete previous_batch;
            previous_batch = 0;
        }

        // Get next batch if available
        if (batch_loader->HasValidationData()) {
            TMVA::Experimental::RTensor<float>* batch = batch_loader->GetValidationBatch();
            previous_batch = batch;
            return batch;
        }
        
        // return empty batch if all events have been used
        return new TMVA::Experimental::RTensor<float>({0,0});
    }

    bool HasTrainData() {
        if (!batch_loader->HasTrainData() && EoF) {
            return false;
        }

        return true;
    }

    bool HasValidationData() {
        if (!batch_loader->HasValidationData()) {
            return false;
        }

        return true;
    }

    void LoadChunks() {
        EoF = false;
        
        // Load chunks untill the end of the file is reached. 
        // Stop loading if a maximum number of chunks is provided
        for (size_t i = 0; ((i < max_chunks) || use_whole_file); i++) {
            LoadChunk();
            if (current_row >= entries) {
                break;
            }
        }    

        batch_loader->DeActivate();
        EoF = true;
    }
};