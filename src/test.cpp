#include "../include/utils.hpp"
#include "../include/pcnn.hpp"
#include <ctime>
#include <unordered_map>
#include <iostream>
#include <array>


#define LOG(msg) utils::logging.log(msg, "TEST")


int main() {


    pcl::SamplingModule sm = pcl::SamplingModule(10);

    sm.print();

    bool keep = false;
    for (int i = 0; i < 28; i++) {
        sm.call(keep);
        if (!sm.is_done()) {
            sm.update(utils::random.getRandomFloat());
        };

        LOG("max " + std::to_string(sm.getMaxValue()));

        if (i == (sm.getSize() + 3)) {
            LOG("resetting...");
            sm.reset();
        };
    };

    return 0;
}
