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

    size_t num_threads, finished_threads = 0, num_rows = 0;
    std::vector<std::thread> threads;
    std::atomic<bool> accept_tasks = true;

    std::queue<std::vector<size_t>> task_queue;
    std::mutex task_lock;
    std::condition_variable task_condition;

    std::queue<TMVA::Experimental::RTensor<float>*> filled_batch_queue;
    std::mutex filled_batch_lock;
    std::condition_variable filled_batch_condition;

    std::queue<TMVA::Experimental::RTensor<float>*> empty_batch_queue;
    std::mutex empty_batch_lock;
    std::condition_variable empty_batch_condition;

    const size_t batch_size, num_columns;

    // Randomize the order of the indices
    void CreateTasks() {
        std::cout << "create tasks" << std::endl;

        // Create a random vector of idx from 0 to chunk_size
        std::vector<size_t> row_order = std::vector<size_t>(num_rows);
        std::iota(row_order.begin(), row_order.end(), 0);
        // std::random_shuffle(row_order.begin(), row_order.end());
        
        // Add the first batch_size elements from the row order to the queue
        // untill the queue is empty
        std::unique_lock<std::mutex> lock(task_lock);
        for(size_t i = 0; i+batch_size-1 < row_order.size(); i+= batch_size) {
            task_queue.push({row_order.begin() + i, row_order.begin() + i + batch_size});
            
        }
        task_lock.unlock();
        task_condition.notify_one();

        std::cout << "created tasks" << std::endl;
    }

    // Fil the batch with rows from the chunk based on the given idx
    void FillBatch(std::vector<size_t> idx, TMVA::Experimental::RTensor<float>* outp) {
        std::cout << "Fill Batch: " << idx[0] << std::endl;
        
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
        std::cout << "Filling done: " << idx[0] << std::endl;
    }

    void idle_loop() {
        std::cout << "THREAD: start thread" << std::endl;

        std::vector<size_t> inp;
        TMVA::Experimental::RTensor<float>* outp;

        while(true) {
            {
                // Wait for new tasks
                std::unique_lock<std::mutex> lock(task_lock);
                task_condition.wait(lock, [this]() {
                    std::cout << "THREAD: task condition: " << task_queue.empty() << std::endl;

                    return !task_queue.empty() || !accept_tasks; });
                
                // Stop thread if all tasks are completed and no more will be added
                if (!accept_tasks && task_queue.empty())
                {
                    finished_threads += 1;
                    task_condition.notify_one();
                    filled_batch_condition.notify_one();
                    return;
                }

                std::cout << "THREAD: getting task: " << task_queue.empty() << " " << accept_tasks << std::endl;
                // get task
                inp = task_queue.front();
                task_queue.pop();  
                lock.unlock();             

                std::cout << "THREAD: got task " << inp[0] << std::endl;

                // Wait for a batch to put the results in
                std::unique_lock<std::mutex> lock_1(empty_batch_lock);
                empty_batch_condition.wait(lock_1, [this]() {
                    std::cout << "THREAD: empty condition: " << empty_batch_queue.empty() << std::endl;
                    
                    return !empty_batch_queue.empty();});
                outp = empty_batch_queue.front();
                empty_batch_queue.pop();
                // lock_1.unlock();

                std::cout << "THREAD: got empty batch" << std::endl;
            }

            FillBatch(inp, outp);
        }
    }

public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    BatchLoader(const size_t batch_size, const size_t num_columns, const size_t num_threads, const size_t num_batches) 
        : batch_size(batch_size), num_columns(num_columns), num_threads(num_threads) 
    {
        for (size_t i = 0; i < num_threads; i++) {
            threads.push_back(std::thread(&BatchLoader::idle_loop, this));
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

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Batch functions
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // return a batch of data
    TMVA::Experimental::RTensor<float>* GetBatch()
    {
        std::cout << "GetBatch" << std::endl;

        std::unique_lock<std::mutex> lock(filled_batch_lock);
        filled_batch_condition.wait(lock, [this]() {
            std::cout << "filled condition: " << filled_batch_queue.empty() << " " << (finished_threads == num_threads) << std::endl;
            
            return !filled_batch_queue.empty() || (finished_threads == num_threads);});
        
        if (filled_batch_queue.empty()) {
            return new TMVA::Experimental::RTensor<float>({0,0});
        }

        TMVA::Experimental::RTensor<float>* res = filled_batch_queue.front();
        filled_batch_queue.pop();
        return res;
    }

    bool HasData() {
        std::cout << "HasData" << std::endl;
        std::unique_lock<std::mutex> lock_1(filled_batch_lock);
        if (!filled_batch_queue.empty()) {
            return true;
        }
        lock_1.unlock();

        std::cout << "no filled batches" << std::endl;

        std::unique_lock<std::mutex> lock_2(task_lock);
        if (!task_queue.empty()) {
            std::cout << "found tasks" << std::endl;
            return true;
        }
        lock_2.unlock();

        std::cout << "no tasks" << std::endl;

        std::cout << "finished_threads: " << finished_threads << std::endl;

        if (finished_threads != num_threads) {
            return true;
        }

        std::cout << "all threads finished" << std::endl;


        if (accept_tasks) {
            return true;
        }

        std::cout << "NO DATA" << std::endl;

        return false;
    }

    void Done() {
        std::unique_lock<std::mutex> lock(task_lock);
        accept_tasks = false;
        lock.unlock();
        task_condition.notify_all();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters and Setters
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Add a batch that can be used to add the results to
    void add_batch(TMVA::Experimental::RTensor<float>* batch) {
        std::cout << "ADDING BATCH" << std:: endl;
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