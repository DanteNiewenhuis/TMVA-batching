#ifndef TMVA_BATCHLOADER
#define TMVA_BATCHLOADER

#include <iostream>
#include <vector>

// Imports for threading
#include <queue>
#include <mutex>
#include <condition_variable>

#include "TMVA/RTensor.hxx"

namespace TMVA {
namespace Experimental {

class BatchLoader
{
private:
    const size_t batch_size, num_columns;

    bool accept_tasks = false;
    TMVA::RandomGenerator<TRandom3> rng;

    // filled batch elements
    std::mutex batch_lock;
    std::condition_variable batch_condition;
    std::queue<TMVA::Experimental::RTensor<float>*> training_batch_queue;

    std::vector<TMVA::Experimental::RTensor<float>*> validation_batches;
    size_t valid_idx = 0;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchLoader(const size_t batch_size, const size_t num_columns) 
        : batch_size(batch_size), num_columns(num_columns) {}

    ~BatchLoader () {}


public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch functions
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // return a batch of data
    TMVA::Experimental::RTensor<float>* GetTrainBatch() {}

    // return a batch of data
    TMVA::Experimental::RTensor<float>* GetValidationBatch() {}

    bool HasTrainData() {}

    bool HasValidationData() {}

    // Activate the threads again to accept new tasks
    void Activate() {}

    // Wait untill all tasks are handled, then join the threads 
    void DeActivate() {}

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Fill the batch with rows from the chunk based on the given idx
    void FillBatch(TMVA::Experimental::RTensor<float>* x_tensor, std::vector<size_t> idx, std::vector<TMVA::Experimental::RTensor<float>*>& batches) {}

    // Add new tasks based on the given x_tensor
    void CreateTrainingBatches(TMVA::Experimental::RTensor<float>* x_tensor, std::vector<size_t> row_order) {}

    // Add new tasks based on the given x_tensor
    void CreateValidationBatches(TMVA::Experimental::RTensor<float>* x_tensor, std::vector<size_t> row_order) {}

    void start_validation() {}
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHLOADER