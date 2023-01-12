#include <iostream>
#include <utility>

#include <tuple>
// #include <functional>

template<typename First, typename... Rest>
class Printer {
private:
    int test = 5;


public:
    void print_inp(First inp) {
        std::cout << "final print: " << test << std::endl;
        std::cout << inp << std::endl;
    }

    void print_inp(First inp, Rest... args) {
        std::cout << "middle print: " << test << std::endl;
        std::cout << inp << std::endl;

        print_inp(std::forward<Rest>(args)...);
    }

    void operator()(First inp, Rest... rest) {
        std::cout << "operator " << inp << std::endl;
    }
};


void template_tester() {
    std::cout << "template tester" << std::endl;

    Printer<float, float> printer;

    printer.print_inp(1, 2);

}


int main() {
    std::cout << "in main" << std::endl;

    template_tester();
}