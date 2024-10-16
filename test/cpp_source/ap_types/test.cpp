#include "ap_float.h"
#include <iostream>
#include <vector>
// int main(int argc, char *argv[]) {
//     ap_float<3, 5, 0> a = 10;
//     // ap_fixed<8, 1> b = 0.9921875;
//     // auto c = b * a;
//     // std::cout << "Hello, World!" << std::endl;
//     std::cout << "a = " << a.to_float() << std::endl;
//     return 0;
// }

int main(int argc, char **argv) {
    // Define the input and output types
    std::vector<ap_float<1, 2, 3>> nums(argc - 1);
    for (int i = 0; i < argc - 1; i++) {
        nums[i] = atof(argv[i + 1]);
    }
    for (int i = 0; i < argc - 1; i++) {
        std::cout << "nums[" << i << "] = " << nums[i].to_float() << std::endl;
    }
}