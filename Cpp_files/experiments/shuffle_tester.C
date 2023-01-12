#include <string>
#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include <random>

size_t incrementer() 
{
    static int i;
    return ++i;
}

void shuffle_tester() 
{
    size_t start = 0, batch_size = 4;

    std::vector<int> foo(10) ; // vector with 100 ints.
    // std::iota (std::begin(foo), std::end(foo), 0); // Fill with 0, 1, ..., 99.

    
    std::generate(foo.begin(), foo.end(), incrementer); 
    std::generate(foo.begin(), foo.end(), [n=-batch_size, batch_size] () mutable {return n += batch_size; }); 
    
    // std::random_shuffle(foo.begin(), foo.end());

    std::cout << "shuffled elements:";
    for (int& x: foo) std::cout << ' ' << x;
    std::cout << '\n';

}