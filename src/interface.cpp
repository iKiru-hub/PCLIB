// interface.cpp
#include "pcnn.hpp"
#include <Eigen/Dense>
#include <iostream>


// Wrapper function to create a new pcNN object
pcNN* create_pcNN(int N, double gain, double offset) {
    return new pcNN(N, gain, offset);
}

// Wrapper function to call the `call` method of pcNN
Eigen::Vector2d call_pcNN(pcNN* model, Eigen::Vector2d x) {
    return model->call(x);
}

// Wrapper function to delete pcNN object
void delete_pcNN(pcNN* model) {
    delete model;
}
