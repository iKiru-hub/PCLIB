#include "../include/utils.hpp"
#include "../include/pcnn.hpp"
#include <ctime>


#define LOG(msg) utils::logging.log(msg, "TEST")


int main() {

    utils::random.setSeed(time(0));

    pcl::SamplingModule sm = pcl::SamplingModule(0.1);
    sm.print();

    int idx = sm.sample();
    LOG("Sampled index: " + std::to_string(idx));

    return 0;
}
