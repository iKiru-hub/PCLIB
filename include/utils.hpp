#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <array>
#include <algorithm>
#include <iterator>


/* declare classes */

class Logger;
class RandomGenerator;



/* LOGGING */


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

    void log_vector(const Eigen::VectorXf &vec,
                    const std::string &src = "MAIN") {

        std::cout << get_datetime() << " | " << src << " | [";
        for (unsigned int i = 0; i < vec.size(); i++) {
            std::cout << vec[i];
            if (i != vec.size() - 1) {
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


/* MISCELLANEOUS */




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

/* @brief: Generalized sigmoid function
 * @param x: input vector
 * @param offset: offset parameter
 * @param gain: gain parameter
 * @param clip: clip parameter
 * @return: sigmoid output : Eigen::Vector2d
 */
inline Eigen::VectorXf generalized_sigmoid(const Eigen::VectorXf& x,
                                           float offset = 1.0f,
                                           float gain = 1.0f,
                                           float clip = 1.0f) {
    // Offset each element by `offset`, apply the gain, and then compute the sigmoid
    std::cout << "offset " << offset << std::endl;
    std::cout << "gain " << gain << std::endl;
    std::cout << "clip " << clip << std::endl;
    Eigen::VectorXf result = 1.0f / (1.0f + \
        (-gain * (x.array() - offset)).exp());

    return (result.array() >= clip).select(result, 0.0f);
}
/* inline Eigen::Vector2f generalized_sigmoid(Eigen::Vector2f x, */
/*                                          double offset = 1.0, */
/*                                          double gain = 1.0, */
/*                                          double clip = 1.0) { */
/*     Eigen::Vector2f offset_vec = Eigen::Vector2f::Constant(offset); */
/*     return (1.0 / (1.0 + (-gain * (x - \ */
/*             offset_vec).array()).exp())).cwiseMin(clip); */
/* } */


inline float cosine_similarity_vec(const Eigen::VectorXf& v1,
                                   const Eigen::VectorXf& v2) {
    return v1.dot(v2) / (v1.norm() * v2.norm());
}

inline Eigen::MatrixXf grahm_matrix(const Eigen::MatrixXf& X) {
    return X * X.transpose();
}

/* @brief: fill a matrix with random values
 * @param M: matrix to fill
 * @param sparsity: sparsity of the matrix
 */
void fill_random(Eigen::MatrixXf &M, float sparsity = 0.0) {

    // make random generator
    RandomGenerator random = RandomGenerator();

    for (int i = 0; i < M.rows(); i++) {
        for (int j = 0; j < M.cols(); j++) {
            if (random.get_random_float() > sparsity) {
                M(i, j) = random.get_random_float();
            }
        }
    }
}

// objects

Logger logging = Logger();
RandomGenerator random = RandomGenerator();

}
