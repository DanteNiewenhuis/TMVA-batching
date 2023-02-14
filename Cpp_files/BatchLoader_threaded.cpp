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

class BatchLoader
{
private:
    TMVA::Experimental::RTensor<float>* x_tensor; 
    size_t num_rows = 0;
    const size_t batch_size, num_columns;

    // thread elements
    std::vector<std::thread> threads;
    size_t num_threads, finished_threads = 0;
    
    // task elements
    std::atomic<bool> accept_tasks = true;
    std::queue<std::vector<size_t>> task_queue;
    std::mutex task_lock;
    std::condition_variable task_condition;

    // filled batch elements
    std::queue<TMVA::Experimental::RTensor<float>*> filled_batch_queue;
    std::mutex filled_batch_lock;
    std::condition_variable filled_batch_condition;

    // empty batch elements
    std::queue<TMVA::Experimental::RTensor<float>*> empty_batch_queue;
    std::mutex empty_batch_lock;
    std::condition_variable empty_batch_condition;

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchLoader(const size_t batch_size, const size_t num_columns, const size_t num_threads, const size_t num_batches) 
        : batch_size(batch_size), num_columns(num_columns), num_threads(num_threads) 
    {
        for (size_t i = 0; i < num_threads; i++) {
            threads.push_back(std::thread(&BatchLoader::IdleLoop, this, i));
        }
        
        for (size_t i = 0; i < num_batches; i++) {
            empty_batch_queue.push(new TMVA::Experimental::RTensor<float>({batch_size, num_columns}));
        }
    }

    ~BatchLoader () {
        for (size_t i = 0; i < num_threads; i++) {
            threads[i].join();
        }
    }


private:
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

    // Randomize the order of the indices
    void CreateTasks() {
        // Create a random vector of idx from 0 to chunk_size
        std::vector<size_t> row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0); // Set values of the elements to 0...num_rows

        std::random_shuffle(row_order.begin(), row_order.end()); // Shuffle the order of idx
        
        // Add the first batch_size elements from the row order to the queue
        // untill the queue is empty
        std::unique_lock<std::mutex> lock(task_lock);
        
        size_t start = 0, end = batch_size;
        while(end < num_rows) {
            // std::vector<size_t> temp = {row_order.begin() + i, row_order.begin() + i + batch_size};
            std::vector<size_t> temp;

            for (size_t i = start; i < end; i++) {
                temp.push_back(row_order[i]);
            }
            task_queue.push(temp);

            start += batch_size;
            end += batch_size;
        }

        task_lock.unlock();
        task_condition.notify_one();
    }

    // Fil the batch with rows from the chunk based on the given idx
    void FillBatch(std::vector<size_t> idx, TMVA::Experimental::RTensor<float>* outp, size_t thread_num) {
        size_t offset;
        for (int i = 0; i < batch_size; i++) {
            std::copy(x_tensor->GetData() + (idx[i]*num_columns), 
                      x_tensor->GetData() + ((idx[i]+1)*num_columns), 
                      outp->GetData() + i*num_columns);
        }

        // std::cout << "Push batch to results" << std::endl;
        // Add outp to result queue
        std::unique_lock<std::mutex> lock(filled_batch_lock);
        filled_batch_queue.push(outp);
        filled_batch_condition.notify_one();
        lock.unlock();
    }

    void IdleLoop(size_t thread_num) {
        std::cout << "start thread: " << thread_num << std::endl;

        std::vector<size_t> inp;
        TMVA::Experimental::RTensor<float>* outp;

        while(true) {
            {
                inp = {};
                outp = 0;
                
                // Wait for new tasks
                std::unique_lock<std::mutex> lock(task_lock);
                task_condition.wait(lock, [this]() {
                    return !task_queue.empty() || !accept_tasks; });
                
                // Stop thread if all tasks are completed and no more will be added
                if (!accept_tasks && task_queue.empty())
                {
                    finished_threads += 1;
                    task_condition.notify_one();
                    filled_batch_condition.notify_one();
                    return;
                }

                if (task_queue.empty()) {
                    std::cout << "IdleLoop " << thread_num << " => ERROR no tasks left in queue" << std::endl;
                }
                // get task
                inp = task_queue.front();
                task_queue.pop();  
                lock.unlock();             


                // Wait for a batch to put the results in
                std::unique_lock<std::mutex> lock_1(empty_batch_lock);
                empty_batch_condition.wait(lock_1, [this]() {                    
                    return !empty_batch_queue.empty();});

                if (empty_batch_queue.empty()) {
                    std::cout << "IdleLoop " << thread_num << " => ERROR no empty batches left in queue" << std::endl;
                }

                outp = empty_batch_queue.front();
                empty_batch_queue.pop();
            }

            std::cout << "start filling: " << thread_num << std::endl;
            FillBatch(inp, outp, thread_num);
        }
    }


public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch functions
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // return a batch of data
    TMVA::Experimental::RTensor<float>* GetBatch()
    {
        std::unique_lock<std::mutex> lock(filled_batch_lock);
        filled_batch_condition.wait(lock, [this]() {            
            return !filled_batch_queue.empty() || (finished_threads == num_threads);});
        
        std::unique_lock<std::mutex> lock_2(empty_batch_lock);
        if (filled_batch_queue.empty()) {
            return new TMVA::Experimental::RTensor<float>({0,0});
        }
        lock_2.unlock();

        TMVA::Experimental::RTensor<float>* res = filled_batch_queue.front();
        filled_batch_queue.pop();
        return res;
    }

    bool HasData() {
        std::unique_lock<std::mutex> lock_1(filled_batch_lock);
        if (!filled_batch_queue.empty())
            return true;
        lock_1.unlock();

        std::unique_lock<std::mutex> lock_2(task_lock);
        if (!task_queue.empty())
            return true;
        lock_2.unlock();

        if (finished_threads != num_threads) 
            return true;

        if (accept_tasks)
            return true;

        return false;
    }

    void Done() {
        std::unique_lock<std::mutex> lock(task_lock);
        accept_tasks = false;
        lock.unlock();
        task_condition.notify_all();
    }

    void wait_for_tasks() {
        std::unique_lock<std::mutex> lock(task_lock);
        task_condition.wait(lock, [this]() {
            return (finished_threads == num_threads); });

    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Add a batch that can be used to add the results to
    void AddBatch(TMVA::Experimental::RTensor<float>* batch) {
        std::unique_lock<std::mutex> lock(empty_batch_lock);
        empty_batch_queue.push(batch);
        lock.unlock();

        empty_batch_condition.notify_one();
    }
    
    
    void SetNumRows(size_t r) {num_rows = r;}

    void SetTensor(TMVA::Experimental::RTensor<float>* x_tensor, const size_t num_rows) {
        this->x_tensor = x_tensor;
        this->num_rows = num_rows;

        CreateTasks();
    }

    std::queue<std::vector<size_t>> GetTaskQueue() {
        return task_queue;
    }


};