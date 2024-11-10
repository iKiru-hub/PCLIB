#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <array>
#include <ctime>
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
    int get_random_element(const std::array<int,
                           N>& container) {
        // Check that the container is not empty
        if (container.empty()) {
            throw std::invalid_argument(
                "Container must not be empty");
        }

        // Create a uniform distribution in the range
        // [0, container.size() - 1]
        std::uniform_int_distribution<> dist(0,
                                container.size() - 1);
        // Use the member random engine
        return container[dist(gen)];
    }

    // @brief function to get a random element from
    // a vector of integers
    int get_random_element_vec(
        const std::vector<int>& container) {
        // Check that the container is not empty
        if (container.empty()) {
            throw std::invalid_argument(
                "Container must not be empty");
        }

        // Create a uniform distribution in the
        // range [0, container.size() - 1]
        std::uniform_int_distribution<> dist(0,
                                    container.size() - 1);
        // Use the random engine
        return container[dist(gen)];
    }

    // Function to get a random float within a specified range
    float get_random_float(float min = 0.0f,
                           float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);  // Use the member random engine
    }

    // random integer in the range [min, max]
    int get_random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

    void set_seed(int seed = -1) {
        if (seed == -1) {
            // Seed with current time
            gen.seed(static_cast<unsigned>(std::time(0)));
        } else {
            // Seed with provided seed
            gen.seed(seed);
        }
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
inline Eigen::VectorXf generalized_sigmoid(
    const Eigen::VectorXf& x,
    float offset = 1.0f,
    float gain = 1.0f,
    float clip = 1.0f) {
    // Offset each element by `offset`, apply the gain,
    // and then compute the sigmoid
    std::cout << "offset " << offset << std::endl;
    std::cout << "gain " << gain << std::endl;
    std::cout << "clip " << clip << std::endl;
    Eigen::VectorXf result = 1.0f / (1.0f + \
        (-gain * (x.array() - offset)).exp());

    return (result.array() >= clip).select(result, 0.0f);
}

// @brief cosine similarity
inline float cosine_similarity_vec(const Eigen::VectorXf& v1,
                                   const Eigen::VectorXf& v2) {
    return v1.dot(v2) / (v1.norm() * v2.norm());
}

// @brief cosine similarity for a matrix
Eigen::MatrixXf cosine_similarity_matrix(
    const Eigen::MatrixXf& matrix) {

    int n = matrix.rows();
    Eigen::MatrixXf similarity_matrix(n, n);

    // Normalize each row to unit norm
    Eigen::MatrixXf normalized_matrix = matrix.rowwise().normalized();

    // Compute the cosine similarity (normalized dot product)
    similarity_matrix = normalized_matrix * normalized_matrix.transpose();

    // take out the diagonal
    similarity_matrix.diagonal().setZero();

    return similarity_matrix;
}

// @brief: calculate the maximum cosine similarity in a column
float max_cosine_similarity_in_column(
    const Eigen::MatrixXf& matrix, int idx) {
    // Compute the cosine similarity matrix
    Eigen::MatrixXf similarity_matrix = cosine_similarity_matrix(matrix);

    // Check that idx is within bounds
    if (idx < 0 || idx >= similarity_matrix.cols()) {
        throw std::out_of_range("Index is out of bounds.");
    }

    // Get the column at the specified index
    Eigen::VectorXf column = similarity_matrix.col(idx);

    // Find the maximum value in the column
    float max_similarity = column.maxCoeff();

    return max_similarity;
}

// @brief calculation of the row to row similarity matrix
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


void test_random_1() {

    RandomGenerator random = RandomGenerator();
    random.set_seed();

    std::vector<int> vec = {10, 1, 2, 3};

    logging.log("size: " + std::to_string(vec.size()));

    int val = random.get_random_element_vec(vec);

    logging.log("Hello, this is: " + std::to_string(val));
}

void test_max_cosine() {
    Eigen::MatrixXf matrix(3, 3);
    matrix << 1, 0, 0,
              1, 0, 1,
              0, 1, 0;

    // print the matrix
    logging.log("Matrix:");
    for (int i = 0; i < matrix.rows(); i++) {
        logging.log_vector(matrix.row(i));
    }

    Eigen::MatrixXf similarity_matrix = cosine_similarity_matrix(matrix);
    float max_sim = max_cosine_similarity_in_column(matrix, 0);
    logging.log("Max cosine similarity: " + std::to_string(max_sim));

    // Print the similarity matrix
    logging.log("Similarity matrix:");
    for (int i = 0; i < similarity_matrix.rows(); i++) {
        logging.log_vector(similarity_matrix.row(i));
    }
}

}
