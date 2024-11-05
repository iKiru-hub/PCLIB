#include <iostream>
#include <Eigen/Dense>
#include "utils.hpp"
#include <unordered_map>
#include <memory>
#include <array>


#define LOG(msg) utils::logging.log(msg, "PCNN")


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


// LEAKY VARIABLE
class LeakyVariableND {
public:
    std::string name;

    /* @brief Call the LeakyVariableND with a 2D input
     * @param x A 2D input to the LeakyVariable */
    void call(const Eigen::VectorXf& x) {

        // Compute dv and update v
        v += (Eigen::VectorXf::Constant(ndim, eq) - v) * tau + x;
    }

    LeakyVariableND(std::string name, float eq, float tau,
                    size_t ndim)
        : name(std::move(name)), eq(eq), tau(1.0f / tau),
        ndim(ndim), v(Eigen::VectorXf::Constant(ndim, eq)) {
        LOG("[+] LeakyVariableND created with name: " + this->name);
    }

    ~LeakyVariableND() {
        LOG("[-] LeakyVariableND destroyed with name: " + name);
    }

    Eigen::VectorXf get_v() const {
        return v;
    }

    void print_v() const {
        std::cout << "v: " << v.transpose() << std::endl;
    }

    std::string str() const {
        return "LeakyVariableND." + name;
    }

    int len() const {
        return ndim;
    }

    void info() const {
        LOG("LeakyVariableND." + name + "(eq=" +
            std::to_string(eq) +
            ", tau=" + std::to_string(tau) + ", ndim=" +
            std::to_string(ndim) + ")");
    }

private:
    float eq;
    float tau;
    size_t ndim;
    Eigen::VectorXf v;
};

class LeakyVariable1D {
public:
    std::string name;

    /* @brief Call the LeakyVariable with an input
     * @param input The input to the LeakyVariable
     * with `ndim` dimensions
    */
    void call(float x = 0.0) {
        v = v + (eq - v) * tau + x;
    }

    LeakyVariable1D(std::string name, float eq,
                    float tau)
        : name(std::move(name)), eq(eq), tau(1.0/tau) {

        v = eq;

        LOG("[+] LeakyVariable1D created with name: " + this->name);
    }

    ~LeakyVariable1D() {
        LOG("[-] LeakyVariable1D destroyed with name: " + name);
    }

    float get_v() {
        return v;
    }

    std::string str() {
        return "LeakyVariable." + name;
    }

    int len() {
        return 1;
    }

    void info() {
        LOG("LeakyVariable." + name + "(eq=" + \
            std::to_string(eq) +
            ", tau=" + std::to_string(tau) + ")");
    }

private:
    float eq;
    float tau;
    float v;
};



// SAMPLING MODULE
class SamplingModule {

public:

    std::string name;

    SamplingModule(std::string name,
                   float speed) : name(name), speed(speed) {
        utils::logging.log("[+] SamplingModule");
    }

    ~SamplingModule() {
        utils::logging.log("[-] SamplingModule");
    }

    void call(bool keep = false) {

        // keep current state
        if (keep) {
            utils::logging.log("-- keep");
            return;
        };

        // all samples have been used
        if (counter == num_samples) {
            keep = true;
            idx = utils::arr_argmax(values);
            velocity = samples[idx];
            utils::logging.log("-- all samples used, picked: " + \
                               std::to_string(idx));
            return;
        };

        idx = sample_idx();
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

    std::string str() {
        return "SamplingModule." + name;
    }

    int len() {
        return num_samples;
    }

    std::array<float, 2> get_velocity() {
        return velocity;
    }

    const int get_idx() {
        return idx;
    }

    const int get_counter() {
        return counter;
    }

    const float get_max_value() {
        if (counter == 0) {
            return 0.0;
        }
        return values[idx];
    }

    void reset() {
        idx = -1;
        for (int i = 0; i < num_samples; i++) {
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
    const std::array<unsigned int, 9> indexes = { 0, 1, 2, 3, 4,
                                          5, 6, 7, 8 };
    unsigned int counter = 0;
    const unsigned int num_samples = 9;
    std::array<float, 9> values = { 0.0 };
    std::array<float, 2> velocity = { 0.0 };

    // variables
    int idx = -1;
    std::array<bool, 9> available_indexes = { true, true, true,
        true, true, true, true, true, true};

    // sample a random index
    int sample_idx() {

        int idx = -1;
        bool found = false;
        while (!found) {
            int i = utils::random.get_random_int(0, num_samples);

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


namespace pcl {


};
