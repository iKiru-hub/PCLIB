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

    /* pcl::test_pcnn(); */
    /* utils::test_random_1(); */
    /* /1* utils::test_max_cosine(); *1/ */
    /* utils::test_connectivity(); */
    /* utils::test_make_position(); */
    pcl::test_randlayer();

    // test tanh
    /* float out = utils::generalized_tanh(-1); */
    /* LOG(std::to_string(out)); */

    return 0;
}

// definitions

