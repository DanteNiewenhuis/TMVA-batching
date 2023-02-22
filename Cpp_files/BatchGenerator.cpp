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
    size_t num_columns, chunk_size, max_chunks, batch_size, current_row=0, entries;

    std::string file_name, tree_name;
    
    BatchLoader* batch_loader;

    std::thread loading_thread;
    bool loading_thread_started = false, initialized = false;

    bool EoF = false, use_whole_file;
    double train_ratio;

    TMVA::Experimental::RTensor<float>* previous_batch = 0;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Functions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    void LoadChunk() 
    {
        // std::cout << "BatchGenerator::LoadChunk => start at row: " << current_row << std::endl;

        TMVA::Experimental::RTensor<float>* x_tensor = new TMVA::Experimental::RTensor<float>({chunk_size, num_columns});

        ChunkLoader<Args...> func((*x_tensor));

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

        current_row += progressed_events;
        batch_loader->AddTasks(x_tensor, passed_events);

        batch_loader->wait_for_tasks();

        std::cout << "BatchGenerator::LoadChunk => Batching Done" << std::endl;
        delete x_tensor;
    }

public:

    BatchGenerator(std::string file_name, std::string tree_name, std::vector<std::string> cols, 
                   std::vector<std::string> filters, size_t chunk_size, size_t batch_size, double train_ratio=1.0, 
                   size_t use_whole_file=true, size_t max_chunks = 1):
        file_name(file_name), tree_name(tree_name), cols(cols), filters(filters), num_columns(cols.size()), 
        chunk_size(chunk_size), batch_size(batch_size), train_ratio(train_ratio), use_whole_file(use_whole_file), max_chunks(max_chunks) {
        
        // get the number of entries in the dataframe
        TFile* f = TFile::Open(file_name.c_str());
        TTree* t = f->Get<TTree>(tree_name.c_str());
        entries = t->GetEntries();

        std::cout << "BatchGenerator => found " << entries << " entries in file." << std::endl;

        size_t num_threads = 1;
        std::cout << "BatchGenerator => train_ratio: " << train_ratio << std::endl;
        batch_loader = new BatchLoader(batch_size, num_columns, num_threads, train_ratio);
    }

    ~BatchGenerator () {
        if (loading_thread_started) {

            loading_thread.join();
        }
    } 

    void init() {
        // needed to make sure nothing crashes when executing init multiple times.
        // TODO: look for better solution

        current_row = 0;

        batch_loader->Activate();
        
        loading_thread = std::thread(&BatchGenerator::LoadChunks, this);
        loading_thread_started = true;

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

        
        for (size_t i = 0; ((i < max_chunks) || (use_whole_file && current_row < entries)); i++) {
            LoadChunk();
            if (current_row >= entries) {
                break;
            }
        }        


        batch_loader->DeActivate();
        EoF = true;
    }
};