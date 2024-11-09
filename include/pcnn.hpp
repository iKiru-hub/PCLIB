#include <iostream>
#include <Eigen/Dense>
#include "utils.hpp"
#include <unordered_map>
#include <memory>
#include <array>


#define LOG(msg) utils::logging.log(msg, "PCLIB")
#define SPACE utils::logging.space


/* PCNN */

class PCLayer {

public:

    PCLayer(int n, float sigma,
            std::array<float, 4> bounds)
        : N(std::pow(n, 2)), n(n), sigma(sigma), bounds(bounds) {

        // Initialize the centers
        centers = Eigen::MatrixXf::Zero(N, 2);

        // Compute the centers
        compute_centers();

        LOG("[+] PCLayer created");
    }

    ~PCLayer() { LOG("[-] PCLayer destroyed"); }

    /* @brief Call the PCLayer with a 2D input and compute
     * the Gaussian distance to the centers
     * @param x A 2D input to the PCLayer
     */
    Eigen::VectorXf call(const Eigen::Vector2f& x) {
        Eigen::VectorXf y = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < N; i++) {
            float dx = x(0) - centers(i, 0);
            float dy = x(1) - centers(i, 1);
            float dist_squared = std::pow(dx, 2) + \
                std::pow(dy, 2);
            y(i) = std::exp(-dist_squared / denom);
        }
        return y;
    }

    std::string str() { return "PCLayer"; }
    int len() { return N; }
    std::string repr() {
        return "PCLayer(n=" + std::to_string(n) + \
            ", sigma=" + std::to_string(sigma) + \
            ", bounds=[" + std::to_string(bounds[0]);
    }

private:

    int N;
    int n;
    float sigma;
    const float denom = 2 * sigma * sigma;
    std::array<float, 4> bounds;
    Eigen::MatrixXf centers;

    void compute_centers() {

        // calculate the spacing between the centers
        // given the number of centers and the bounds
        float x_spacing = (bounds[1] - bounds[0]) / (n - 1);
        float y_spacing = (bounds[3] - bounds[2]) / (n - 1);

        // Compute the centers
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                centers(i * n + j, 0) = bounds[0] + \
                    i * x_spacing;
                centers(i * n + j, 1) = bounds[2] + \
                    j * y_spacing;
            }
        }
    }

};


class PCNN {
public:
    PCNN(int N, int Nj, float gain, float offset,
         float threshold, float rep_threshold,
         float rec_threshold,
         int num_neighbors, float trace_tau,
         PCLayer xfilter, std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors), trace_tau(trace_tau),
        xfilter(xfilter), name(name) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        C = Eigen::MatrixXf::Zero(N, N);
        mask = Eigen::VectorXf::Zero(N);
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
        num_pc = 0;
        free_indexes = Eigen::VectorXi::LinSpaced(N + 1, 0, N);

        // make ff matrix random
        utils::fill_random(Wff, 0.5);
    }

    Eigen::VectorXf call(const Eigen::Vector2f& x,
                         const bool frozen = false) {

        // pass the input through the filter layer
        Eigen::VectorXf x_filtered = xfilter.call(x);

        LOG("[+] PCNN.call");
        utils::logging.log_vector(x_filtered);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = Wff * x_filtered + pre_x;

        LOG("pre-activation");
        utils::logging.log_vector(u);

        u = utils::generalized_sigmoid(u, offset, gain, 0.01);

        // update the trace
        trace = (1 - trace_tau) * trace + trace_tau * u;

        LOG("post-activation");
        utils::logging.log_vector(u);

        // update model
        if (!frozen) {
            update();
        }

        return u;
    }

    void reset() {
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
    }

    // Getters
    int len() const { return N; }
    std::string str() const { return "PCNN." + name; }
    std::string repr() const {
        return "PCNN(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
            std::to_string(rep_threshold) + \
            std::to_string(rec_threshold) + \
            std::to_string(num_neighbors) + \
            std::to_string(trace_tau) + ")";
    }
    Eigen::VectorXf representation() const { return u; }

private:
    // parameters
    const int N;
    const int Nj;
    const float gain;
    const float offset;
    const float threshold;
    const float rep_threshold;
    const float rec_threshold;
    const int num_neighbors;
    const float trace_tau;
    const std::string name;

    PCLayer xfilter;

    // variables
    Eigen::MatrixXf Wff;
    Eigen::MatrixXf Wffbackup;
    Eigen::MatrixXf Wrec;
    Eigen::MatrixXf C;
    Eigen::VectorXf mask;
    Eigen::VectorXi learnt_indexes;
    Eigen::VectorXi free_indexes;
    int num_pc;
    Eigen::VectorXf u;
    Eigen::VectorXf trace;
    Eigen::VectorXf pre_x;

    void update() {}

    // @brief Quantify the indexes
    void quantify_indexes(float threshold = 0.99) {
        // the learned indexes are the indexes of the
        // neurons that have a sum of weights greater
        // than 0.99.
        // learnt: [i1, -1, i3, ...]
        // free: [-1, i2, -1, ...]
        for (int i = 0; i < N; i++) {
            if (Wff.row(i).sum() > threshold) {
                learnt_indexes.push_back(i);
                free_indexes.erase(i);
            }
        }

    }

};



// LEAKY VARIABLE
class LeakyVariableND {
public:

    LeakyVariableND(std::string name, float eq, float tau,
                    size_t ndim)
        : name(std::move(name)), eq(eq), tau(1.0f / tau),
        ndim(ndim), v(Eigen::VectorXf::Constant(ndim, eq)) {
        LOG("[+] LeakyVariableND created with name: " + this->name);
    }

    ~LeakyVariableND() {
        LOG("[-] LeakyVariableND destroyed with name: " + name);
    }

    /* @brief Call the LeakyVariableND with a 2D input
     * @param x A 2D input to the LeakyVariable */
    void call(const Eigen::VectorXf& x) {

        // Compute dv and update v
        v += (Eigen::VectorXf::Constant(ndim, eq) - v) * tau + x;
    }


    Eigen::VectorXf get_v() const { return v; }
    void print_v() const {
        std::cout << "v: " << v.transpose() << std::endl; }
    std::string str() const { return "LeakyVariableND." + name; }
    int len() const { return ndim; }

    void repr() const {
        LOG("LeakyVariableND." + name + "(eq=" +
            std::to_string(eq) +
            ", tau=" + std::to_string(tau) + ", ndim=" +
            std::to_string(ndim) + ")");
    }

private:
    float eq;
    float tau;
    size_t ndim;
    std::string name;
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

        LOG("[+] LeakyVariable1D created with name: " + \
            this->name);
    }

    ~LeakyVariable1D() {
        LOG("[-] LeakyVariable1D destroyed with name: " + name);
    }

    float get_v() { return v; }
    std::string str() { return "LeakyVariable." + name; }
    int len() { return 1; }
    void repr() {
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

    ~SamplingModule() { utils::logging.log("[-] SamplingModule"); }

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

    void update(float score) { values[idx] = score; }
    bool is_done() { return counter == num_samples; }
    std::string str() { return "SamplingModule." + name; }
    int len() { return num_samples; }
    std::array<float, 2> get_velocity() { return velocity; }
    const int get_idx() { return idx; }
    const int get_counter() { return counter; }

    const float get_max_value() {
        if (counter == 0) { return 0.0; }
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

void test_layer() {

    std::array<float, 4> bounds = {0.0, 1.0, 0.0, 1.0};

    PCLayer layer = PCLayer(3, 0.1, bounds);
    LOG(layer.str());
    LOG(std::to_string(layer.len()));

};


void testSampling() {

    pcl::SamplingModule sm = pcl::SamplingModule(10);

    sm.str();

    bool keep = false;
    for (int i = 0; i < 28; i++) {
        sm.call(keep);
        if (!sm.is_done()) {
            sm.update(utils::random.get_random_float());
        };

        if (i == (sm.get_size() + 3)) {
            LOG("resetting...");
            sm.reset();
        };
    };
};


void test_leaky() {

    SPACE("#---#");

    LOG("Testing LeakyVariable...");


    SPACE("#---#");
}


void test_pcnn() {

    SPACE("#---#");

    LOG("Testing PCNN...");

    int n = 3;
    int Nj = std::pow(n, 2);

    PCLayer xfilter = PCLayer(n, 0.1, {0.0, 1.0, 0.0, 1.0});
    LOG(xfilter.str());

    PCNN model = PCNN(3, Nj, 10., 0.5, 0.1,
                      0.1, 8, 0.1, xfilter);

    LOG(model.str());

    Eigen::Vector2f x = {0.5, 0.5};
    Eigen::VectorXf y = model.call(x);

    LOG("output:");
    utils::logging.log_vector(y);

    LOG("");

    SPACE("#---#");
}

};
