#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <iostream>
#include <vector>
#include <thread>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "ChunkLoader.cpp" // TODO: change to TMVA thing
#include "BatchLoader.cpp" // TODO: change to TMVA thing

namespace TMVA {
namespace Experimental {

template<typename... Args>
class BatchGenerator 
{
private:
    TMVA::RandomGenerator<TRandom3> rng;

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

    std::vector<std::vector<size_t>> training_idxs;
    std::vector<std::vector<size_t>> validation_idxs;

    std::vector<size_t> vec_sizes;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Functions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Load chunk_size rows of the given RDataFrame into a RTensor.
    // After, the chunk of data is split into batches of data.
    void LoadChunk(size_t current_chunk) {}

    void createIdxs(size_t current_chunk, size_t progressed_events) {}

public:

    BatchGenerator(std::string file_name, std::string tree_name, std::vector<std::string> cols, 
                   std::vector<std::string> filters, size_t chunk_size, size_t batch_size, std::vector<size_t> vec_sizes = {}, double validation_split=0.0, 
                   size_t max_chunks = 0, size_t num_columns = 0):
        file_name(file_name), tree_name(tree_name), cols(cols), filters(filters), num_columns(num_columns), 
        chunk_size(chunk_size), batch_size(batch_size), vec_sizes(vec_sizes), validation_split(validation_split), max_chunks(max_chunks) {}

    ~BatchGenerator () {} 

    void StopLoading() {}

    void init() {}

    // Returns the next batch of data if available. 
    // Returns empty RTensor otherwise.
    TMVA::Experimental::RTensor<float>* GetTrainBatch() {}

    // Returns the next batch of data if available. 
    // Returns empty RTensor otherwise.
    TMVA::Experimental::RTensor<float>* GetValidationBatch() {}

    bool HasTrainData() {}

    bool HasValidationData() {}

    void LoadChunks() {}

    void start_validation() {}
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR