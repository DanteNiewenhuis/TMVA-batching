#include <iostream>
#include <utility>
#include <tuple>
#include <vector>

template<typename... Args>
void print_inp(Args... args) {
    std::cout << "printing" << std::endl;
    ((std::cout << args << ' '), ...);
    std::cout << std::endl;
}

void template_tester_2() {
    print_inp(1, 2, 4);

}