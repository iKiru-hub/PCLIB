#include "../include/utils.hpp"
#include "../include/pcnn.hpp"
#include <ctime>
#include <Eigen/Dense>
#include <unordered_map>
#include <iostream>
#include <array>

#define LOG(msg) utils::logging.log(msg, "TEST")
#define SPACE utils::logging.space


// MAIN

int main() {

    /* testSampling(); */
    /* testLeaky(); */
    /* pcl::test_layer(); */
    pcl::test_pcnn();
    /* utils::test_random_1(); */
    /* utils::test_max_cosine(); */

    return 0;
}

// definitions

