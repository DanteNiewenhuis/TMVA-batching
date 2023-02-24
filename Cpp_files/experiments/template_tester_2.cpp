#include <iostream>
#include <utility>
#include <tuple>
#include <vector>


void print_inp(int first) {
    std::cout << "INT final" << std::endl;
    std::cout << "val: " << first << std::endl;
}

template<typename First>
void print_inp(First first) {
    std::cout << "NOT INT final" << std::endl;
    std::cout << "val: " << first << std::endl;
}


template<typename... Args>
void print_inp(int first, Args... args) {
    std::cout << "INT middle" << std::endl;
    std::cout << "val: " << first << std::endl;

    print_inp(std::forward<Args>(args)...);
}

template<typename First, typename... Args>
void print_inp(First first, Args... args) {
    std::cout << "NOT INT middle" << std::endl;
    std::cout << "val: " << first << std::endl;

    print_inp(std::forward<Args>(args)...);
}


void template_tester_2() {
    print_inp<float, float, float, float>(1, 2.1, 4, 6.5);

}