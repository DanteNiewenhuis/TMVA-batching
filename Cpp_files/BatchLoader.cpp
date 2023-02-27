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

// Struct for the Tasks given to the threads,
// A Task states which rows of the given tensor should be used for a batch
struct Task {
    TMVA::Experimental::RTensor<float>* x_tensor; 
    bool is_validation;
    std::vector<size_t> idx;
};

class BatchLoader
{
private:
    const size_t batch_size, num_columns;
    double validation_split;

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
    BatchLoader(const size_t batch_size, const size_t num_columns, const size_t num_threads, double validation_split=0.7) 
        : batch_size(batch_size), num_columns(num_columns), num_threads(num_threads), validation_split(validation_split)
    {
    }

    ~BatchLoader () {
        StopThreads();
    }

private:

    void StartThreads() {

        // make sure all threads are joined before starting new threads
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i].join();
        }

        for (size_t i = 0; i < num_threads; i++) {
            threads.push_back(std::thread(&BatchLoader::IdleLoop, this, i));
        }
    }

    void StopThreads() {
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i].join();
        }
        threads.clear();
    }

    // Fil the batch with rows from the chunk based on the given idx
    void FillBatch(Task task, size_t thread_num) {
        
        // // 
        // TMVA::Experimental::RTensor<float>* x_tensor = task.x_tensor;
        // std::vector<size_t> idx = task.idx;

        active_threads += 1;
        
        // Copy rows from x_tensor to the new batch
        TMVA::Experimental::RTensor<float>* batch = new TMVA::Experimental::RTensor<float>({batch_size, num_columns});
        for (int i = 0; i < batch_size; i++) {
            std::copy(task.x_tensor->GetData() + (task.idx[i]*num_columns), 
                      task.x_tensor->GetData() + ((task.idx[i]+1)*num_columns), 
                      batch->GetData() + i*num_columns);
        }

        if (task.is_validation){
            std::unique_lock<std::mutex> lock(validation_batch_lock);
            validation_batch_queue.push(batch);
            validation_batch_condition.notify_one();
            lock.unlock();
        }
        else{
            std::unique_lock<std::mutex> lock(training_batch_lock);
            training_batch_queue.push(batch);
            training_batch_condition.notify_one();
            lock.unlock();
        }

        active_threads -= 1;
        task_condition.notify_all();
    }

    // Wait untill a new Task is available, then execute it
    void IdleLoop(size_t thread_num) {

        Task task;
        while(true) {
            {
                // Wait for new tasks
                std::unique_lock<std::mutex> lock(task_lock);
                task_condition.wait(lock, [this]() {return !task_queue.empty() || !accept_tasks; });
                
                // Stop thread if all tasks are completed and no more will be added
                if (task_queue.empty() && !accept_tasks){
                    task_condition.notify_one();
                    return;
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

    // Activate the threads again to accept new tasks
    void Activate() {
        std::unique_lock<std::mutex> lock(task_lock);
        accept_tasks = true;
        lock.unlock();
        task_condition.notify_all();

        StartThreads();
    }

    // Wait untill all tasks are handled, then join the threads 
    void DeActivate() {

        std::unique_lock<std::mutex> lock(task_lock);
        accept_tasks = false;
        lock.unlock();
        task_condition.notify_all();
        training_batch_condition.notify_one();

        StopThreads();
    }

    // Wait untill all tasks are. Do not join the threads. 
    void wait_for_tasks() {
        std::unique_lock<std::mutex> lock(task_lock);
        task_condition.wait(lock, [this]() {
            return task_queue.empty() && active_threads == 0; });

    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Add new tasks based on the given x_tensor
    void AddTasks(TMVA::Experimental::RTensor<float>* x_tensor, const size_t num_rows) {

        // Calculate the number of batches that will be used for validation
        size_t num_validation = (num_rows / batch_size) * validation_split;

        // create a vector of integers from 0 to chunk_size and shuffle it
        std::vector<size_t> row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0); // Set values of the elements to 0...num_rows
        std::random_shuffle(row_order.begin(), row_order.end()); // Shuffle the order of idx

        
        std::unique_lock<std::mutex> lock(task_lock);
        
        // Create tasks of batch_size untill all idx are used 
        size_t task_num = 0;
        for(size_t start = 0; (start + batch_size) <= num_rows; start += batch_size) {
            Task task;
            task.x_tensor = x_tensor;
            
            for (size_t i = start; i < (start + batch_size); i++) {
                task.idx.push_back(row_order[i]);
            }

            task.is_validation = num_validation > task_num++; 

            task_queue.push(task);
        }

        task_lock.unlock();
        task_condition.notify_all();
    }

    std::queue<Task> GetTaskQueue() {
        return task_queue;
    }
};