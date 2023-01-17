#include <tuple>
#include <iostream>
#include <array>
#include <utility>

// debugging aid
template <typename T>
void print_sequence(T t) {
  std::cout << t << std::endl;
}


// debugging aid
template <typename T, typename... Rest>
void print_sequence(T t, Rest... rest) {
  std::cout << t << std::endl;
  
  print_sequence(std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest>
class Class_A {

public:
    Class_A() {}

    void operator()(First first) {
        std::cout << first << std::endl;
    }

    void operator()(First first, Rest... rest) {
        std::cout << first << std::endl;

        (*this)(std::forward<Rest>(rest)...);
    }
};

void packing_test() 
{   
    Class_A<int, float> printer;

    printer(2, 3.5);
}