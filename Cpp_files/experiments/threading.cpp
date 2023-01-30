#include <iostream>
#include <vector>
#include <cstdlib>
#include <thread> 
#include "TMVA/RTensor.hxx"

void fill(std::vector<int>* lst) {

    for (int i = 0; i < 5000000; i++) {

        lst->push_back(rand());
    }

}

void threading() {
    std::vector<int> lst = {};
    
    std::thread t (fill, &lst);

    // for (auto i : lst) {
    //     std::cout << i << std::endl;
    // }

    t.join();


    std::cout << lst[0] << std::endl;
    std::cout << lst.size() << std::endl;

    TMVA::Experimental::RTensor<float> f({1, 2});
}

int main() {
    threading();
}