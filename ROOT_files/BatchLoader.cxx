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
        : batch_size(batch_size), num_columns(num_columns)
    {
        rng = TMVA::RandomGenerator<TRandom3>(0);
    }

    ~BatchLoader () 
    {}


public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch functions
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // return a batch of data
    TMVA::Experimental::RTensor<float>* GetTrainBatch()
    {
        std::unique_lock<std::mutex> lock(batch_lock);
        batch_condition.wait(lock, [this]() {     
            return !training_batch_queue.empty() || !accept_tasks;});
        
        if (training_batch_queue.empty()) {
            return new TMVA::Experimental::RTensor<float>({0,0});
        }

        TMVA::Experimental::RTensor<float>* res = training_batch_queue.front();
        training_batch_queue.pop();
        return res;
    }

    // return a batch of data
    TMVA::Experimental::RTensor<float>* GetValidationBatch()
    {
        return validation_batches[valid_idx++];
    }

    bool HasTrainData() {
        std::unique_lock<std::mutex> lock(batch_lock);
        if (!training_batch_queue.empty() || accept_tasks)
            return true;
        lock.unlock();

        return false;
    }

    bool HasValidationData() {
        return valid_idx < validation_batches.size();
    }

    // Activate the threads again to accept new tasks
    void Activate() {
        accept_tasks = true;
        batch_condition.notify_all();
    }

    // Wait untill all tasks are handled, then join the threads 
    void DeActivate() {
        accept_tasks = false;
        batch_condition.notify_one();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Fill the batch with rows from the chunk based on the given idx
    void FillBatch(TMVA::Experimental::RTensor<float>* x_tensor, std::vector<size_t> idx, std::vector<TMVA::Experimental::RTensor<float>*>& batches) { 
        // Copy rows from x_tensor to the new batch
        TMVA::Experimental::RTensor<float>* batch = new TMVA::Experimental::RTensor<float>({batch_size, num_columns});
        for (int i = 0; i < batch_size; i++) {
            std::copy(x_tensor->GetData() + (idx[i]*num_columns), 
                      x_tensor->GetData() + ((idx[i]+1)*num_columns), 
                      batch->GetData() + i*num_columns);
        }

        batches.push_back(batch);
    }

    // Add new tasks based on the given x_tensor
    void CreateTrainingBatches(TMVA::Experimental::RTensor<float>* x_tensor, std::vector<size_t> row_order) {
        std::shuffle(row_order.begin(), row_order.end(),rng); // Shuffle the order of idx

        std::vector<TMVA::Experimental::RTensor<float>*> batches;
        
        // Create tasks of batch_size untill all idx are used 
        for(size_t start = 0; (start + batch_size) <= row_order.size(); start += batch_size) {
            
            std::vector<size_t> idx;

            for (size_t i = start; i < (start + batch_size); i++) {
                idx.push_back(row_order[i]);
            }

            FillBatch(x_tensor, idx, batches);
        }

        std::unique_lock<std::mutex> lock(batch_lock);
        for (size_t i = 0; i < batches.size(); i++) {
            training_batch_queue.push(batches[i]);
        }

        lock.unlock();
        batch_condition.notify_one();
    }

    // Add new tasks based on the given x_tensor
    void CreateValidationBatches(TMVA::Experimental::RTensor<float>* x_tensor, std::vector<size_t> row_order) {
        std::vector<TMVA::Experimental::RTensor<float>*> batches;
        
        // Create tasks of batch_size untill all idx are used 
        for(size_t start = 0; (start + batch_size) <= row_order.size(); start += batch_size) {
            
            std::vector<size_t> idx;

            for (size_t i = start; i < (start + batch_size); i++) {
                idx.push_back(row_order[i]);
            }

            FillBatch(x_tensor, idx, batches);
        }

        for (size_t i = 0; i < batches.size(); i++) {
            validation_batches.push_back(batches[i]);
        }
    }

    void start_validation() {
        valid_idx = 0;
    }
};

} // namespace Experimental
} // namespace TMVA