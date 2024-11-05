#include "../include/utils.hpp"
#include "../include/pcnn.hpp"
#include <ctime>
#include <Eigen/Dense>
#include <unordered_map>
#include <iostream>
#include <array>

#define LOG(msg) utils::logging.log(msg, "TEST")
#define SPACE utils::logging.space


// init
void testSampling();
void testLeaky();

// MAIN

int main() {

    /* testSampling(); */
    testLeaky();

    return 0;
}

// definitions

void testSampling() {

    /* pcl::SamplingModule sm = pcl::SamplingModule(10); */

    /* sm.print(); */

    /* bool keep = false; */
    /* for (int i = 0; i < 28; i++) { */
    /*     sm.call(keep); */
    /*     if (!sm.is_done()) { */
    /*         sm.update(utils::random.getRandomFloat()); */
    /*     }; */

    /*     if (i == (sm.getSize() + 3)) { */
    /*         LOG("resetting..."); */
    /*         sm.reset(); */
    /*     }; */
    /* }; */
};



void testLeaky() {

    SPACE("#---#");

    LOG("Testing LeakyVariable...");


    SPACE("#---#");
}
