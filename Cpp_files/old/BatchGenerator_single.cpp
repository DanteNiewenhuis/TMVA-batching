#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"

#include "ChunkLoader.cpp"
#include "BatchLoader.cpp"


template<typename... Args>
class BatchGenerator 
{
private:
    std::vector<std::string> cols, filters;
    size_t num_columns, chunk_size, max_chunks, batch_size, current_row=0, entries;

    std::string file_name, tree_name;

    TMVA::Experimental::RTensor<float>* x_tensor;
    BatchLoader* batch_loader;

public:

    BatchGenerator(std::string file_name, std::string tree_name, std::vector<std::string> cols, std::vector<std::string> filters, 
                   size_t chunk_size, size_t batch_size, size_t max_chunks):
        file_name(file_name), tree_name(tree_name), cols(cols), filters(filters), num_columns(cols.size()), chunk_size(chunk_size), 
        max_chunks(max_chunks), batch_size(batch_size) {
        
        // get the number of entries in the dataframe
        TFile* f = TFile::Open(file_name.c_str());
        TTree* t = f->Get<TTree>(tree_name.c_str());
        entries = t->GetEntries();

        std::cout << "found " << entries << " entries in file." << std::endl;

        x_tensor = new TMVA::Experimental::RTensor<float>({chunk_size, num_columns});
        
        std::cout << "batch_size: " << batch_size << std::endl;
        batch_loader = new BatchLoader(batch_size, num_columns);
    }

    void load_chunk() 
    {
        std::cout << "load_chunk starting at row: " << current_row << std::endl;
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
            myReport->Print();
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

        // std::cout << (*x_tensor) << std::endl;
        // std::cout << progressed_events << std::endl;
        // std::cout << passed_events << std::endl;

        batch_loader->SetTensor(x_tensor, passed_events);
        current_row += progressed_events;
    }

    TMVA::Experimental::RTensor<float>* get_batch()
    {   
        // get the next batch if available
        if (batch_loader->HasData()) {
            // std::cout << "batch available" << std::endl;
            return (*batch_loader)();
        }

        // load new chunk
        if (current_row < entries) {
            load_chunk();
            return get_batch();
        }
        
        // return empty batch if all events have been used
        auto tensor = new TMVA::Experimental::RTensor<float>({0,0});
        return tensor;
    }

    bool HasData() {
        if (current_row <= entries) {
            return true;
        }

        return false;
    }

    void init() {}
};