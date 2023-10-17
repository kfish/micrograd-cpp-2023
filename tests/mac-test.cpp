#include <iostream>

#include "mac.h"

int main(int argc, char *argv[])
{
    std::vector<int> vec_a = {1, 2, 3};
    std::vector<int> vec_b = {4, 5, 6};
    std::array<int, 3> arr_a = {1, 2, 3};
    std::array<int, 3> arr_b = {4, 5, 6};

    std::cout << "Vector multiply-accumulate: " << mac(vec_a, vec_b) << std::endl;
    std::cout << "Array multiply-accumulate: " << mac(arr_a, arr_b) << std::endl;
}

