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
    void LoadChunk() 

public:
    BatchGenerator(std::string file_name, std::string tree_name, std::vector<std::string> cols, 
                   std::vector<std::string> filters, size_t chunk_size, size_t batch_size, std::vector<size_t> vec_sizes = {}, double validation_split=0.0, 
                   size_t max_chunks = 0, size_t num_columns = 0);
    ~BatchGenerator();
    void StopLoading();
    TMVA::Experimental::RTensor<float>* GetTrainBatch();
    TMVA::Experimental::RTensor<float>* GetValidationBatch();
    
    bool HasTrainData();
    bool HasValidationData();
    void LoadChunks();
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR