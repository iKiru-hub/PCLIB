#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <array>
#include <algorithm>
#include <iterator>



/* MISCELLANEOUS */


std::string get_datetime() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%X", &tstruct);
    return buf;
}


class Logger {

public:
    Logger() {
        std::cout << get_datetime() << " [+] Logger" << std::endl;
    }

    ~Logger() {
        std::cout << get_datetime() << " [-] Logger" << std::endl;
    }

    void log(const std::string &msg,
             const std::string &src = "MAIN") {
        std::cout << get_datetime() << " | " << src \
            << " | " << msg << std::endl;
    }

    void space(const ::std::string &symbol = ".") {
        std::cout << symbol << std::endl;
    }

    template <std::size_t N>
    void log_arr(const std::array<float, N> &arr,
                  const std::string &src = "MAIN") {

        std::cout << get_datetime() << " | " << src << " | [";
        for (unsigned int i = 0; i < arr.size(); i++) {
            std::cout << arr[i];
            if (i != arr.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    template <std::size_t N>
    void log_arr_bool(const std::array<bool, N> &arr,
                      const std::string &src = "MAIN") {

        std::cout << get_datetime() << " | " << src << " | [";
        for (unsigned int i = 0; i < arr.size(); i++) {
            std::cout << arr[i];
            if (i != arr.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
};




/* RANDOM GENERATORS */


class RandomGenerator {
public:
    // Constructor with an optional seed for deterministic randomness
    RandomGenerator(int seed = 0) : gen(std::mt19937(seed)) {
        std::cout << get_datetime() << " [+] RandomGenerator" << std::endl;
    }

    ~RandomGenerator() {
        std::cout << get_datetime() << " [-] RandomGenerator" << std::endl;
    }

    // Template function to get a random element from an array
    template <std::size_t N>
    int get_random_element(const std::array<int, N>& container) {
        // Check that the container is not empty
        if (container.empty()) {
            throw std::invalid_argument("Container must not be empty");
        }

        // Create a uniform distribution in the range [0, container.size() - 1]
        std::uniform_int_distribution<> dist(0, container.size() - 1);
        return container[dist(gen)];  // Use the member random engine
    }

    // Function to get a random float within a specified range
    float get_random_float(float min = 0.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);  // Use the member random engine
    }

    // random integer in the range [min, max]
    int get_random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

    void set_seed(int seed) {
        gen.seed(seed);
    }

private:
    std::mt19937 gen;  // Mersenne Twister random engine
};


/* NAMESPACE */

namespace utils {


//
template <std::size_t N>
int arr_argmax(std::array<float, N> arr) {

    // Get iterator to the maximum element
    auto max_it = std::max_element(arr.begin(), arr.end());

    // Calculate the index
    return std::distance(arr.begin(), max_it);
}

inline Eigen::Vector2d generalized_sigmoid(Eigen::Vector2d x,
                                         double offset = 1.0,
                                         double gain = 1.0,
                                         double clip = 1.0) {
    Eigen::Vector2d offset_vec = Eigen::Vector2d::Constant(offset);
    return (1.0 / (1.0 + (-gain * (x - offset_vec).array()).exp())).cwiseMin(clip);
}



// objects

Logger logging = Logger();
RandomGenerator random = RandomGenerator();

}
