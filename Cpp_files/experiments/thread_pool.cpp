#include <iostream>
#include <vector>
#include <cstdlib>
#include <thread> 
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>


#include <unistd.h>
#include "TMVA/RTensor.hxx"


class ThreadPool 
{
private:
    size_t num_threads, finished_threads = 0;

    std::vector<std::thread> threads;
    std::condition_variable condition;
    std::atomic<bool> accept_tasks = true;

    std::queue<std::vector<int>> task_queue;
    std::mutex task_lock;

    std::queue<std::vector<int>*> result_queue;
    std::mutex result_lock;

    std::queue<std::vector<int>*> batches_queue;
    std::mutex batches_lock; 

public:

    ThreadPool(size_t num_threads): num_threads(num_threads) {
        for (size_t i = 0; i < num_threads; i++) {
            threads.push_back(std::thread(&ThreadPool::thread_loop, this));
        }

        for (size_t i = 0; i < 3; i++) {
            batches_queue.push(new std::vector<int>({0,0,0}));
        }    
    }

    ~ThreadPool() {
        for (size_t i = 0; i < num_threads; i++) {
            threads[i].join();
        }
    }

    // Add a task to the queue of tasks
    void add_task(std::vector<int> task) {
        std::unique_lock<std::mutex> lock(task_lock);

        task_queue.push(task);

        lock.unlock();
        condition.notify_one();
    }

    // Add a batch that can be used to add the results to
    void add_batch(std::vector<int>* batch) {
        std::unique_lock<std::mutex> lock(batches_lock);

        batches_queue.push(batch);

        lock.unlock();
        condition.notify_one();
    }

    // Process the given input and put the results in the given output
    void process_task(std::vector<int> inp, std::vector<int>* outp) {
        std::cout << "start process" << std::endl;

        usleep(10000);

        // fill output
        for (size_t i = 0; i < inp.size(); i++) {
            std::cout << inp[i] << " ";
            outp->at(i) = inp[i]*2;
        }

        // Add result to the result queue
        std::unique_lock<std::mutex> lock(result_lock);
        result_queue.push(outp);
        condition.notify_one();
        lock.unlock();
    }

    // A loop for the thread that is executed when the thread is idle. 
    void thread_loop() {
        std::cout << "start thread" << std::endl;
        std::vector<int> inp;
        std::vector<int>* outp;
        while (true) {
            {
                // Wait for new tasks
                std::unique_lock<std::mutex> lock(task_lock);
                condition.wait(lock, [this]() {return !task_queue.empty() || !accept_tasks; });
                if (!accept_tasks && task_queue.empty())
                {
                    finished_threads += 1;
                    condition.notify_one();

                    //finish the thread loop and let it join in the main thread.
                    return;
                }
                
                // get task
                inp = task_queue.front();
                task_queue.pop();                

                // Wait for a batch to put the results in
                std::unique_lock<std::mutex> lock_1(batches_lock);
                condition.wait(lock_1, [this]() {return !batches_queue.empty();});
                outp = batches_queue.front();
                batches_queue.pop();
            }

            // Process the given in and output
            process_task(inp, outp);
        }
    }

    // Notify the threads that no more new tasks are coming
    void done() {
        std::unique_lock<std::mutex> lock(task_lock);
        accept_tasks = false;
        lock.unlock();
        condition.notify_all();
    }

    // Get the next result when available
    std::vector<int>* get_result() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(result_lock);
                condition.wait(lock, [this]() {
                    return !result_queue.empty();});
                std::vector<int>* res = result_queue.front();
                result_queue.pop();
                return res;
            }
        }
    }

    // Check if all tasks are completed. 
    bool tasks_completed() {
        return result_queue.empty() && 
               task_queue.empty() && 
               finished_threads == num_threads && 
               !accept_tasks;
    }

};

int main() {
    // create the batches
    std::vector<std::vector<int>*> batches;

    std::queue<std::vector<int>*> queue;
    // for (size_t i = 0; i < 3; i++) {
    //     std::vector<int>* t = new std::vector<int>({0,0,0});
    //     batches.push_back(t);
    // }


    // for (size_t i = 0; i < 3; i++) {
    //     queue.push(new std::vector<int>({0,0,0}));
    // }
    
    // std::cout << queue.front() << std::endl;
    // queue.pop();    
    // std::cout << queue.front() << std::endl;
    // queue.pop();    
    // std::cout << queue.front() << std::endl;
    // queue.pop();

    // std::cout << &batches[0] << std::endl;
    // std::cout << &batches[1] << std::endl;
    // std::cout << &batches[2] << std::endl;

    // Create threadpool
    ThreadPool pool(1);
    
    // Create Tasks
    for (int i = 0; i < 5*3; i += 3) {   
        pool.add_task({i+1,i+2,i+3});
    }
    
    pool.done();

    usleep(1000000);

    std::vector<int>* res;
    while (!pool.tasks_completed()) {
        res = pool.get_result();
        
        std::cout << "result: " << res << res->size() << std::endl;


        std::cout << "result: ";
        for (size_t i = 0; i < res->size(); i++) {
            std::cout << res->at(i) << " ";
        }
        std::cout << std::endl;

        usleep(1000000);

        pool.add_batch(res);
    }

    std::cout << "end" << std::endl;
}