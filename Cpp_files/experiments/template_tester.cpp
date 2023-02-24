#include <iostream>
#include <utility>

#include <tuple>
// #include <functional>

template<typename First, typename... Rest>
class Printer {

public:

    void operator()(First first) {
        std::cout << first << std::endl;
    }

    void operator()(First first, Rest... rest) {
        std::cout << first << std::endl;
        (*this)(std::forward<Rest>(rest)...);
    }

    void operator()(int first) {
        std::cout << "INT" << std::endl;

        std::cout << first << std::endl;
        (*this)(std::forward<Rest>(rest)...);
    }

    void operator()(int first, Rest... rest) {
        std::cout << "INT" << std::endl;

        std::cout << first << std::endl;
        (*this)(std::forward<Rest>(rest)...);
    }
};

template<typename... Args>
void wrapper(float x, float y) {

    Printer<Args...> printer;

    printer(3, 5);
}

void template_tester() {
    std::cout << "template tester" << std::endl;

    wrapper<int, int>(2.1, 3.76);

}


int main() {
    std::cout << "in main" << std::endl;

    template_tester();
}