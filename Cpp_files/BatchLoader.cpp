#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

// Threading imports
#include <thread> 
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "TMVA/RTensor.hxx"

struct Task {
    TMVA::Experimental::RTensor<float>* x_tensor; 
    std::vector<size_t> idx;
};

class BatchLoader
{
private:
    const size_t batch_size, num_columns;
    double train_ratio;

    // thread elements
    std::vector<std::thread> threads;
    size_t num_threads, active_threads = 0;
    
    // task elements
    std::atomic<bool> accept_tasks = true;
    std::queue<Task> task_queue;
    std::mutex task_lock;
    std::condition_variable task_condition;

    // filled batch elements
    std::queue<TMVA::Experimental::RTensor<float>*> training_batch_queue;
    std::mutex training_batch_lock;
    std::condition_variable training_batch_condition;

    // filled batch elements
    std::queue<TMVA::Experimental::RTensor<float>*> validation_batch_queue;
    std::mutex validation_batch_lock;
    std::condition_variable validation_batch_condition;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchLoader(const size_t batch_size, const size_t num_columns, const size_t num_threads, double train_ratio=0.7) 
        : batch_size(batch_size), num_columns(num_columns), num_threads(num_threads), train_ratio(train_ratio)
    {
    }

    ~BatchLoader () {
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i].join();
        }
    }

    void init() {
        StartThreads();
    }

private:

    void StartThreads() {
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i].join();
        }

        for (size_t i = 0; i < num_threads; i++) {
            threads.push_back(std::thread(&BatchLoader::IdleLoop, this, i));
        }
    }

    std::vector<size_t> getRange(size_t start, size_t end) {
        if (start > end) {
            return {};
        }

        std::vector<size_t> res;
        for (size_t i = start; i < end; i++) {
            res.push_back(i);
        }

        return res;
    }

    bool IsTrainBatch() {
        double val = (double)rand() / RAND_MAX;

        return val < train_ratio;
    }

    // Fil the batch with rows from the chunk based on the given idx
    void FillBatch(Task task, size_t thread_num) {
        
        TMVA::Experimental::RTensor<float>* x_tensor = task.x_tensor;
        std::vector<size_t> idx = task.idx;

        active_threads += 1;
        TMVA::Experimental::RTensor<float>* outp = new TMVA::Experimental::RTensor<float>({batch_size, num_columns});
        
        size_t offset;
        for (int i = 0; i < batch_size; i++) {
            std::copy(x_tensor->GetData() + (idx[i]*num_columns), 
                      x_tensor->GetData() + ((idx[i]+1)*num_columns), 
                      outp->GetData() + i*num_columns);
        }

        bool is_train = IsTrainBatch();
        if (is_train){
            std::unique_lock<std::mutex> lock(training_batch_lock);
            training_batch_queue.push(outp);
            training_batch_condition.notify_one();
            lock.unlock();
        }
        else{
            std::unique_lock<std::mutex> lock(validation_batch_lock);
            validation_batch_queue.push(outp);
            validation_batch_condition.notify_one();
            lock.unlock();
        }

        active_threads -= 1;
        task_condition.notify_all();
    }

    void IdleLoop(size_t thread_num) {
        // std::cout << "BatchLoader::IdleLoop => start thread: " << thread_num << std::endl;

        Task task;

        while(true) {
            {
                // Wait for new tasks
                std::unique_lock<std::mutex> lock(task_lock);
                task_condition.wait(lock, [this]() {
                    return !task_queue.empty() || !accept_tasks; });
                
                // Stop thread if all tasks are completed and no more will be added
                if (task_queue.empty() && !accept_tasks){
                    // std::cout << "BatchLoader::IdleLoop " << thread_num << " => Done tasks" << std::endl;
                    task_condition.notify_one();
                    return;
                }

                if (task_queue.empty()) {
                    std::cout << "BatchLoader::IdleLoop " << thread_num << " => ERROR no tasks left in queue" << std::endl;
                }
                // get task
                task = task_queue.front();
                task_queue.pop();  
                lock.unlock();
            }

            FillBatch(task, thread_num);
        }
    }


public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch functions
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // return a batch of data
    TMVA::Experimental::RTensor<float>* GetTrainBatch()
    {
        std::unique_lock<std::mutex> lock(training_batch_lock);
        training_batch_condition.wait(lock, [this]() {     
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
        std::unique_lock<std::mutex> lock(validation_batch_lock);
        validation_batch_condition.wait(lock, [this]() {            
            return !validation_batch_queue.empty();});
        
        if (validation_batch_queue.empty()) {
            return new TMVA::Experimental::RTensor<float>({0,0});
        }

        TMVA::Experimental::RTensor<float>* res = validation_batch_queue.front();
        validation_batch_queue.pop();
        return res;
    }

    bool HasTrainData() {
        std::unique_lock<std::mutex> lock_1(training_batch_lock);
        if (!training_batch_queue.empty() || accept_tasks)
            return true;
        lock_1.unlock();

        return false;
    }

    bool HasValidationData() {
        std::unique_lock<std::mutex> lock_1(validation_batch_lock);
        if (!validation_batch_queue.empty())
            return true;
        lock_1.unlock();
        return false;
    }

    void Activate() {
        std::unique_lock<std::mutex> lock(task_lock);
        accept_tasks = true;
        lock.unlock();
        task_condition.notify_all();

        StartThreads();
    }

    void DeActivate() {
        // std::cout << "BatchLoader::DeActivate => start" << std::endl;

        std::unique_lock<std::mutex> lock(task_lock);
        accept_tasks = false;
        lock.unlock();
        task_condition.notify_all();
        training_batch_condition.notify_one();
        // std::cout << "BatchLoader::DeActivate => end" << std::endl;
    }

    void wait_for_tasks() {
        // std::cout << "BatchLoader::wait_for_tasks => start" << std::endl;
        std::unique_lock<std::mutex> lock(task_lock);
        task_condition.wait(lock, [this]() {
            return task_queue.empty() && active_threads == 0; });

    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Randomize the order of the indices
    void AddTasks(TMVA::Experimental::RTensor<float>* x_tensor, const size_t num_rows) {
        // std::cout << "BatchLoader::AddTasks => start" << std::endl;

        // Add a random vector of idx from 0 to chunk_size
        std::vector<size_t> row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0); // Set values of the elements to 0...num_rows

        // std::random_shuffle(row_order.begin(), row_order.end()); // Shuffle the order of idx
        std::unique_lock<std::mutex> lock(task_lock);
        
        
        // Create tasks of batch_size untill all idx are used 
        for(size_t start = 0; (start + batch_size) <= num_rows; start += batch_size) {
            Task task;
            task.x_tensor = x_tensor;
            
            for (size_t i = start; i < (start + batch_size); i++) {
                task.idx.push_back(row_order[i]);
            }

            task_queue.push(task);
        }

        // std::cout << "BatchLoader::AddTasks => done" << std::endl;
        task_lock.unlock();
        task_condition.notify_all();
    }

    std::queue<Task> GetTaskQueue() {
        return task_queue;
    }


};