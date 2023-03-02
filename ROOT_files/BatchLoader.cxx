namespace TMVA {
namespace Experimental {

class BatchLoader
{
private:
    const size_t batch_size, num_columns;
    double validation_split;

    // thread elements
    std::vector<std::thread> threads;
    size_t num_threads, active_threads = 0;

    bool accept_tasks = false;
    TMVA::RandomGenerator<TRandom3> rng;

    // filled batch elements
    std::queue<TMVA::Experimental::RTensor<float>*> training_batch_queue;
    std::queue<TMVA::Experimental::RTensor<float>*> validation_batch_queue;
    std::mutex batch_lock;
    std::condition_variable batch_condition;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchLoader(const size_t batch_size, const size_t num_columns, double validation_split=0.0) 
        : batch_size(batch_size), num_columns(num_columns), validation_split(validation_split)
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
        std::unique_lock<std::mutex> lock(batch_lock);
        batch_condition.wait(lock, [this]() {            
            return !validation_batch_queue.empty();});
        
        if (validation_batch_queue.empty()) {
            return new TMVA::Experimental::RTensor<float>({0,0});
        }

        TMVA::Experimental::RTensor<float>* res = validation_batch_queue.front();
        validation_batch_queue.pop();
        return res;
    }

    bool HasTrainData() {
        std::unique_lock<std::mutex> lock(batch_lock);
        if (!training_batch_queue.empty() || accept_tasks)
            return true;
        lock.unlock();

        return false;
    }

    bool HasValidationData() {
        std::unique_lock<std::mutex> lock(batch_lock);
        if (!validation_batch_queue.empty())
            return true;
        lock.unlock();

        return false;
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
    void CreateBatches(TMVA::Experimental::RTensor<float>* x_tensor, const size_t num_rows) {

        // Calculate the number of batches that will be used for validation

        // create a vector of integers from 0 to chunk_size and shuffle it
        std::vector<size_t> row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0); // Set values of the elements to 0...num_rows
        
        std::shuffle(row_order.begin(), row_order.end(),rng); // Shuffle the order of idx

        std::vector<TMVA::Experimental::RTensor<float>*> batches;
        
        // Create tasks of batch_size untill all idx are used 
        for(size_t start = 0; (start + batch_size) <= num_rows; start += batch_size) {
            
            std::vector<size_t> idx;

            for (size_t i = start; i < (start + batch_size); i++) {
                idx.push_back(row_order[i]);
            }

            FillBatch(x_tensor, idx, batches);
        }

        // Push the batches to the queues
        size_t num_validation = batches.size() * validation_split;
        std::unique_lock<std::mutex> lock(batch_lock);
        for (size_t i = 0; i < batches.size(); i++) {
            if (i < num_validation)
                validation_batch_queue.push(batches[i]);
            else
                training_batch_queue.push(batches[i]);
        }

        lock.unlock();
        batch_condition.notify_one();
    }
};

} // namespace Experimental
} // namespace TMVA