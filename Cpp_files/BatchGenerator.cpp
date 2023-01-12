#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"

#include "ChunkLoader.cpp"
#include "BatchLoader.cpp"

class BatchGenerator 
{
private:
    std::vector<std::string> cols;
    size_t num_columns, chunk_size, max_chunks, batch_size, current_row=0, entries;

    string file_name, tree_name;

    bool EoF = false;

    TMVA::Experimental::RTensor<float>* x_tensor;
    BatchLoader* batch_loader;

public:

    BatchGenerator(string file_name, string tree_name, std::vector<std::string> cols, size_t chunk_size, size_t batch_size, size_t max_chunks):
        file_name(file_name), tree_name(tree_name), cols(cols), num_columns(cols.size()), chunk_size(chunk_size), max_chunks(max_chunks), batch_size(batch_size) {
        
        // get the number of entries in the dataframe
        TFile* f = TFile::Open(file_name.c_str());
        TTree* t = f->Get<TTree>(tree_name.c_str());
        entries = t->GetEntries();

        std::cout << entries << std::endl;

        x_tensor = new TMVA::Experimental::RTensor<float>({chunk_size, num_columns});
        batch_loader = new BatchLoader(batch_size, num_columns);
    }

    void load_chunk() 
    {
        std::cout << "load_chunk " << current_row << std::endl;
        ChunkLoader<float, std::make_index_sequence<20>> func((*x_tensor), num_columns, chunk_size);

        // Create DataFrame        
        long long start_l = current_row;
        long long end_l = start_l + chunk_size;
        ROOT::Internal::RDF::RDatasetSpec x_spec = ROOT::Internal::RDF::RDatasetSpec(tree_name, 
                                                file_name, {start_l, std::numeric_limits<Long64_t>::max()});
        ROOT::RDataFrame x_rdf = ROOT::Internal::RDF::MakeDataFrameFromSpec(x_spec);

        // add filter

        auto myCount = x_rdf.Range(0, chunk_size).Count();

        x_rdf.Range(0, chunk_size).Foreach(func, cols);

        size_t loaded_size = myCount.GetValue();
        if (loaded_size < chunk_size) {
            EoF = true;
        }

        batch_loader->SetTensor(x_tensor, loaded_size);
        current_row += chunk_size;
    }

    TMVA::Experimental::RTensor<float>* get_batch()
    {
        if (batch_loader->HasData()) {
            return (*batch_loader)();
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