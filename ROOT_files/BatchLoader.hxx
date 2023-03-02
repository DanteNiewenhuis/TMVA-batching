#ifndef TMVA_BATCHLOADER
#define TMVA_BATCHLOADER

#include <iostream>
#include <vector>

// Imports for threading
#include <thread> 
#include <queue>
#include <mutex>
#include <condition_variable>

#include "TMVA/RTensor.hxx"

namespace TMVA {
namespace Experimental {

class BatchLoader
{


public:
    BatchLoader(const size_t batch_size, const size_t num_columns, double validation_split=0.0)
    TMVA::Experimental::RTensor<float>* GetTrainBatch();
    TMVA::Experimental::RTensor<float>* GetValidationBatch();
    bool HasTrainData();
    bool HasValidationData();
    void Activate();
    void DeActivate();
    void FillBatch(TMVA::Experimental::RTensor<float>* x_tensor, std::vector<size_t> idx, std::vector<TMVA::Experimental::RTensor<float>*>& batches);
    void CreateBatches(TMVA::Experimental::RTensor<float>* x_tensor, const size_t num_rows);
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHLOADER