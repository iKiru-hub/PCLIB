#pragma once
#include <iostream>
#include <Eigen/Dense>
#include "utils.hpp"
#include <unordered_map>


/* UTILS */
inline Eigen::Vector2d generalized_sigmoid(Eigen::Vector2d x,
                                         double offset = 1.0,
                                         double gain = 1.0,
                                         double clip = 1.0) {
    Eigen::Vector2d offset_vec = Eigen::Vector2d::Constant(offset);
    return (1.0 / (1.0 + (-gain * (x - offset_vec).array()).exp())).cwiseMin(clip);
}


class pcNN {
public:
    pcNN(int N, double gain, double offset)
        : N(N), gain_(gain), offset_(offset) {
        W_ = Eigen::Matrix2d::Zero();
        C_ = Eigen::Matrix2d::Zero();
        mask_ = Eigen::Matrix2d::Zero();
        u_ = Eigen::Vector2d::Zero();
    }

    Eigen::Vector2d call(const Eigen::Vector2d& x) {
        u_ = generalized_sigmoid(W_ * u_ + x, offset_, gain_, 0.01);
        return u_;
    }

    // Getters
    int get_N() const { return N; }
    double get_gain() const { return gain_; }
    double get_offset() const { return offset_; }
    const Eigen::Matrix2d& get_W() const { return W_; }
    const Eigen::Matrix2d& get_C() const { return C_; }
    const Eigen::Matrix2d& get_mask() const { return mask_; }
    const Eigen::Vector2d& get_u() const { return u_; }

    // Setters
    void set_W(const Eigen::Matrix2d& W) { W_ = W; }
    void set_C(const Eigen::Matrix2d& C) { C_ = C; }
    void set_mask(const Eigen::Matrix2d& mask) { mask_ = mask; }
    void set_u(const Eigen::Vector2d& u) { u_ = u; }

    // class representation
    /* void print() { */
    /*     std::cout << "pcNN(" << N << ", " << gain_ << ", " << offset_ << ")" << std::endl; */
    /* } */

private:
    const int N;
    const double gain_;
    const double offset_;
    Eigen::Matrix2d W_;
    Eigen::Matrix2d C_;
    Eigen::Matrix2d mask_;
    Eigen::Vector2d u_;
};


namespace pcl {

class SamplingModule {

public:

    SamplingModule(float speed) : speed(speed) {
        utils::logging.log("[+] SamplingModule");
    }

    ~SamplingModule() {
        utils::logging.log("[-] SamplingModule");
    }

    void print() {
        utils::logging.log("SamplingModule(speed=" + \
                std::to_string(speed) + ")");
    }

    void call(bool keep) {

        // keep current state
        if (keep) {
            utils::logging.log("-- keep");
            return;
        };

        // all samples have been used
        if (counter == num_samples) {
            keep = true;
            idx = utils::arrArgMax(values);
            velocity = samples[idx];
            utils::logging.log("-- all samples used, picked: " + std::to_string(idx));
            return;
        };

        idx = sampleIdx();
        available_indexes[idx] = false;
        velocity = samples[idx];
        utils::logging.log("-- sampled: " + std::to_string(idx));
        return;
    }

    void update(float score) {
        values[idx] = score;
    }

    bool is_done() {
        return counter == num_samples;
    }

    std::array<float, 2> getVelocity() {
        return velocity;
    }

    const char getIdx() {
        return idx;
    }

    const char getSize() {
        return num_samples;
    }

    const float getMaxValue() {
        if (counter == 0) {
            return 0.0;
        }
        return values[idx];
    }

    void reset() {
        idx = -1;
        for (char i = 0; i < num_samples; i++) {
            available_indexes[i] = true;
        }
        values = { 0.0 };
        counter = 0;
    }

private:

    // parameters
    const float speed;
    const std::array<float, 2> samples[9] = {
        {-1.0, 1.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {-1.0, 0.0},
        {0.0, 0.0},
        {1.0, 0.0},
        {-1.0, -1.0},
        {0.0, -1.0},
        {1.0, -1.0}
    };
    const std::array<unsigned char, 9> indexes = { 0, 1, 2, 3, 4,
                                          5, 6, 7, 8 };
    unsigned char counter = 0;
    const unsigned char num_samples = 9;
    std::array<float, 9> values = { 0.0 };
    std::array<float, 2> velocity = { 0.0 };

    // variables
    int idx = -1;
    std::array<bool, 9> available_indexes = { true, true, true,
        true, true, true, true, true, true};

    // sample a random index
    int sampleIdx() {

        int idx = -1;
        bool found = false;
        while (!found) {
            int i = utils::random.getRandomInt(0, num_samples);

            // Check if the index is available
            if (available_indexes[i]) {
                idx = i;  // Set idx to the found index
                found = true;  // Mark as found
            };
        };
        counter++;
        return idx;
    }


};



};
