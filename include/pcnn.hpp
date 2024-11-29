#include <iostream>
#include <Eigen/Dense>
#include "utils.hpp"
#include <unordered_map>
#include <memory>
#include <array>
#include <tuple>


/* #define LOG(msg) utils::logging.log(msg, "PCLIB") */
#define SPACE utils::logging.space

// blank log function
void LOG(const std::string& msg) {}

// DEBUGGING logs

bool DEBUGGING = false;

void set_debug(bool flag) {
    DEBUGGING = flag;
}

void DEBUG(const std::string& msg) {
    if (DEBUGGING) {
        std::cout << "[DEBUG] " << msg << std::endl;
    }
}


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
    Eigen::MatrixXf get_centers() { return centers; }
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
         float clip_min, float threshold,
         float rep_threshold,
         float rec_threshold,
         int num_neighbors, float trace_tau,
         PCLayer xfilter, std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors),
        trace_tau(trace_tau),
        xfilter(xfilter), name(name) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        connectivity = Eigen::MatrixXf::Zero(N, N);
        mask = Eigen::VectorXf::Zero(N);
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        delta_wff = 0.0;
        x_filtered = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
        cell_count = 0;

        ach = 1.0;

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            free_indexes.push_back(i);
        }
        fixed_indexes = {};
    }

    Eigen::VectorXf call(const Eigen::Vector2f& x,
                         const bool frozen = false,
                         const bool traced = true) {

        // pass the input through the filter layer
        x_filtered = xfilter.call(x);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = Wff * x_filtered + pre_x;

        u = utils::generalized_sigmoid_vec(u, offset, gain, clip_min);

        // update the trace
        if (traced) {
            trace = (1 - trace_tau) * trace + trace_tau * u;
        }

        // update model
        /* if (!frozen) { */
        /*     update(x_filtered); */
        /* } */

        return u;
    }

    // @brief update the model
    void update() {

        make_indexes();

        // exit: a fixed neuron is above threshold
        if (check_fixed_indexes() != -1) {
            DEBUG("!Fixed index above threshold");
           return void();
        };

        // exit: there are no free neurons
        if (free_indexes.size() == 0) {
            DEBUG("!No free neurons");
            return void();
        };

        // pick new index
        int idx = utils::random.get_random_element_vec(
                                        free_indexes);
        DEBUG("Picked index: " + std::to_string(idx));

        // determine weight update
        Eigen::VectorXf dw = x_filtered - Wff.row(idx).transpose();
        delta_wff = dw.norm();

        if (delta_wff > 0.0) {

            DEBUG("delta_wff: " + std::to_string(delta_wff));

            // update weights
            Wff.row(idx) += dw.transpose();

            // calculate the similarity among the rows
            float similarity = \
                utils::max_cosine_similarity_in_rows(
                    Wff, idx);

            // check repulsion (similarity) level
            if (similarity > (rep_threshold * ach)) {
                Wff.row(idx) = Wffbackup.row(idx);
                return void();
            }

            // update count and backup
            cell_count++;
            Wffbackup.row(idx) = Wff.row(idx);

            // update recurrent connections
            update_recurrent();

        }

    }

   Eigen::VectorXf fwd_ext(const Eigen::Vector2f& x) {
        return call(x, true, false);
    }

   Eigen::VectorXf fwd_int(const Eigen::VectorXf& a) {
        return Wrec * a + pre_x;
    }

    void reset() {
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
    }

    // Getters
    int len() const { return cell_count; }
    int get_size() const { return N; }
    std::string str() const { return "PCNN." + name; }
    std::string repr() const {
        return "PCNN(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
            std::to_string(rec_threshold) + \
            std::to_string(num_neighbors) + \
            std::to_string(trace_tau) + ")";
    }
    Eigen::VectorXf representation() const { return u; }
    Eigen::MatrixXf get_wff() const { return Wff; }
    Eigen::MatrixXf get_wrec() const { return Wrec; }
    Eigen::MatrixXf get_connectivity() const { return connectivity; }
    Eigen::VectorXf get_trace() const { return trace; }
    float get_delta_update() const { return delta_wff; }

    Eigen::MatrixXf get_centers() {
        return utils::calc_centers_from_layer(
            Wff, xfilter.get_centers());
    }

    // @brief modulate the density of new PCs
    void ach_modulation(float ach) {
        this->ach = ach;
    }

private:
    // parameters
    const int N;
    const int Nj;
    const float gain;
    const float offset;
    const float clip_min;
    const float threshold;
    const float rep_threshold;
    const float rec_threshold;
    const int num_neighbors;
    const float trace_tau;
    const std::string name;

    float ach;

    PCLayer xfilter;

    // variables
    Eigen::MatrixXf Wff;
    Eigen::MatrixXf Wffbackup;
    float delta_wff;
    Eigen::MatrixXf Wrec;
    Eigen::MatrixXf connectivity;
    /* Eigen::MatrixXf centers; */
    Eigen::VectorXf mask;
    std::vector<int> fixed_indexes;
    std::vector<int> free_indexes;
    int cell_count;
    Eigen::VectorXf u;
    Eigen::VectorXf x_filtered;
    Eigen::VectorXf trace;
    Eigen::VectorXf pre_x;

    // @brief check if one of the fixed neurons
    int check_fixed_indexes() {

        // if there are no fixed neurons, return -1
        if (fixed_indexes.size() == 0) {
            return -1;
        };

        // loop through the fixed indexes's `u` value
        // and return the index with the highest value
        int max_idx = -1;
        float max_u = 0.0;
        for (int i = 0; i < fixed_indexes.size(); i++) {
            if (u(fixed_indexes[i]) > max_u) {
                max_u = u(fixed_indexes[i]);
                max_idx = fixed_indexes[i];
            };
        };

        if (max_u < (threshold * ach) ) { return -1; }
        else {
            DEBUG("Fixed index above threshold: " + \
                std::to_string(max_u) + \
                " [" + std::to_string(threshold) + "]");
            return max_idx; };
    }

    // @brief Quantify the indexes.
    void make_indexes() {

        free_indexes.clear();
        for (int i = 0; i < N; i++) {
            if (Wff.row(i).sum() > threshold) {
                fixed_indexes.push_back(i);
            } else {
                free_indexes.push_back(i);
            }
        }
    }

    // @brief calculate the recurrent connections
    void update_recurrent() {
        // connectivity matrix
        connectivity = utils::connectivity_matrix(
            Wff, rec_threshold
        );

        // similarity
        Wrec = utils::cosine_similarity_matrix(Wff);

        // weights
        Wrec = Wrec.cwiseProduct(connectivity);
    }

};


/* MODULATION MODULES */

// leaky variable

class LeakyVariableND {
public:

    LeakyVariableND(std::string name, float eq, float tau,
                    int ndim, float min_v = 0.0)
        : name(std::move(name)), tau(1.0f / tau), eq_base(eq),
        ndim(ndim), min_v(min_v),
        v(Eigen::VectorXf::Constant(ndim, eq)),
        eq(Eigen::VectorXf::Constant(ndim, eq)){

        LOG("[+] LeakyVariableND created with name: " + this->name);
    }

    ~LeakyVariableND() {
        LOG("[-] LeakyVariableND destroyed with name: " + name);
    }

    /* @brief Call the LeakyVariableND with a 2D input
     * @param x A 2D input to the LeakyVariable */
    Eigen::VectorXf call(const Eigen::VectorXf x,
                         const bool simulate = false) {

        // simulate
        if (simulate) {
            Eigen::VectorXf z = v + (eq - v) * tau + x;
            for (int i = 0; i < ndim; i++) {
                if (z(i) < min_v) {
                    z(i) = min_v;
                }
            }
            return z;
        }

        // Compute dv and update v
        v += (eq - v) * tau + x;
        return v;
    }

    Eigen::VectorXf get_v() const { return v; }
    void print_v() const {
        std::cout << "v: " << v.transpose() << std::endl; }
    std::string str() const { return "LeakyVariableND." + name; }
    int len() const { return ndim; }
    std::string repr() const {
        return "LeakyVariableND." + name + "(eq=" + \
            std::to_string(eq_base) + ", tau=" + std::to_string(tau) + \
            ", ndim=" + std::to_string(ndim) + ")";
    }
    std::string get_name() { return name; }
    void set_eq(const Eigen::VectorXf& eq) { this->eq = eq; }
    void reset() {
        for (int i = 0; i < ndim; i++) {
            v(i) = eq(i);
        }
    }

private:
    const float tau;
    const int ndim;
    const float min_v;
    const float eq_base;
    std::string name;
    Eigen::VectorXf v;
    Eigen::VectorXf eq;
};


class LeakyVariable1D {
public:
    std::string name;

    /* @brief Call the LeakyVariable with an input
     * @param input The input to the LeakyVariable
     * with `ndim` dimensions
    */
    float call(float x = 0.0,
               bool simulate = false) {

        // simulate
        if (simulate) {
            float z = v + (eq - v) * tau + x;
            if (z < min_v) {
                z = min_v;
            }
            return z;
        }

        v = v + (eq - v) * tau + x;

        if (v < min_v) {
            v = min_v;
        }
        return v;
    }

    LeakyVariable1D(std::string name, float eq,
                    float tau, float min_v = 0.0)
        : name(std::move(name)), eq(eq), tau(1.0/tau),
        v(eq), min_v(min_v){

        LOG("[+] LeakyVariable1D created with name: " + \
            this->name);
    }

    ~LeakyVariable1D() {
        LOG("[-] LeakyVariable1D destroyed with name: " + name);
    }

    float get_v() { return v; }
    std::string str() { return "LeakyVariable." + name; }
    std::string repr() {
        return "LeakyVariable." + name + "(eq=" + \
            std::to_string(eq) + ", tau=" + std::to_string(tau) + ")";
    }
    int len() { return 1; }
    std::string get_name() { return name; }
    void set_eq(float eq) { this->eq = eq; }
    void reset() { v = eq; }

private:
    const float min_v;
    const float tau;
    float v;
    float eq;
};


// density modulation

class DensityMod {

public:

    DensityMod(std::array<float, 5> weights,
               float theta):
        weights(weights), theta(theta), baseline(theta) {}

    ~DensityMod() {}

    float call(const std::array<float, 5>& x) {
        dtheta = 0.0;
        for (size_t i = 0; i < 5; i++) {
            dtheta += x[i] * weights[i];
        }
        theta = baseline + utils::generalized_tanh(
            dtheta, 0.0, 1.0);
        return theta;
    }
    std::string str() { return "DensityMod"; }
    float get_value() { return theta; }

private:
    std::array<float, 5> weights;
    float baseline;
    float theta;
    float dtheta;
};


/* ACTION SAMPLING MODULE */


class ActionSampling2D {

public:

    std::string name;

    ActionSampling2D(std::string name,
                   float speed) : name(name)
                {
        update_actions(speed);
        /* utils::logging.log("[+] ActionSampling2D." + name); */
    }

    ~ActionSampling2D() {
        /* utils::logging.log("[-] ActionSampling2D." + name); */
    }

    std::tuple<std::array<float, 2>, bool, int> call(bool keep = false) {

        // Keep current state
        if (keep) {
            utils::logging.log("-- keep");
            return std::make_tuple(velocity, false, idx);
        }

        // All samples have been used
        if (counter == num_samples) {

            // try to sample a zero index
            int zero_idx = sample_zero_idx();

            if (zero_idx != -1) {
                idx = zero_idx;
            } else {
                // Get the index of the maximum value
                idx = utils::arr_argmax(values);
            }

            velocity = samples[idx];
            return std::make_tuple(velocity, true, idx);
        }

        // Sample a new index
        idx = sample_idx();
        available_indexes[idx] = false;
        velocity = samples[idx];

        return std::make_tuple(velocity, false, idx);
    }

    void update(float score = 0.0f) { values[idx] = score; }
    bool is_done() { return counter == num_samples; }
    std::string str() { return "ActionSampling2D." + name; }
    std::string repr() { return "ActionSampling2D." + name; }
    int len() { return num_samples; }
    const int get_idx() { return idx; }
    const int get_counter() { return counter; }

    const float get_max_value() {
        if (counter == 0) { return 0.0; }
        return values[idx];
    }

    // @brief get values for the samples
    const std::array<float, 9> get_values() { return values; }

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
    std::array<float, 2> samples[9] = {
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
    int idx = -1;

    // @brief variables
    /* int idx = -1; */
    std::array<bool, 9> available_indexes = { true, true, true,
        true, true, true, true, true, true};

    // @brief sample a random index
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

    // @brief make a set of how values equal to zero
    // and return a random index from it
    int sample_zero_idx() {

        std::vector<int> zero_indexes;
        for (size_t i = 0; i < num_samples; i++) {
            if (values[i] == 0.0) {
                zero_indexes.push_back(i);
            }
        }

        if (zero_indexes.size() > 1) {
            return utils::random.get_random_element_vec(zero_indexes);
        }

        return -1;
    }

    // @brief update the actions given a speed
    void update_actions(float speed) {

        for (size_t i = 0; i < num_samples; i++) {
            float dx = samples[i][0];
            float dy = samples[i][1];
            float scale = speed / std::sqrt(2.0f);
            if (dx == 0.0 && dy == 0.0) {
                continue;
            } else if (dx == 0.0) {
                dy *= speed;
            } else if (dy == 0.0) {
                dx *= speed;
            } else {
                // speed / sqrt(2)
                dx *= scale;
                dy *= scale;
            }
            samples[i] = {dx, dy};
        }
    }

};


struct TwoLayerNetwork {

    // @brief forward an input through the network
    // @return a tuple: (float, array<float, 2>)
    std::tuple<float, std::array<float, 2>>
    call(const std::array<float, 5>& x) {
        hidden = {0.0, 0.0};
        output = 0.0;

        // hidden layer
        for (size_t i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                hidden[i] += x[j] * w_hidden[i][j];
            }
        }

        // output layer
        for (size_t i = 0; i < 2; i++) {
            output += hidden[i] * w_output[i];
        }

        return std::make_tuple(output, hidden);
    }

    TwoLayerNetwork(std::array<std::array<float, 2>, 5> w_hidden,
                   std::array<float, 2> w_output)
        : w_hidden(w_hidden), w_output(w_output) {}
    ~TwoLayerNetwork() {}
    std::string str() { return "TwoLayerNetwork"; }

private:

    //  matrix 5x2
    const std::array<std::array<float, 2>, 5> w_hidden;
    const std::array<float, 2> w_output;
    std::array<float, 2> hidden;
    float output;
};


struct OneLayerNetwork {

    // @brief forward an input through the network
    // @return tuple(float, array<float, 2>)
    std::tuple<float, std::array<float, 5>>
    call(const std::array<float, 5> x) {

        output = 0.0;
        z = {};
        for (size_t i = 0; i < 5; i++) {
            output += x[i] * weights[i];
            z[i] = x[i] * weights[i];
        }
        return std::make_tuple(output, z);
    }

    OneLayerNetwork(std::array<float, 5> weights)
        : weights(weights) {
        z = {};
    }
    ~OneLayerNetwork() {}
    std::string str() { return "OneLayerNetwork"; }
    std::array<float, 5> get_weights() { return weights; }

private:

    const std::array<float, 5> weights;
    float output;
    std::array<float, 5> z;

};


/* EXPERIENCE MODULE */


class ExperienceModule {

};


namespace pcl {

void test_layer() {

    std::array<float, 4> bounds = {0.0, 1.0, 0.0, 1.0};

    PCLayer layer = PCLayer(3, 0.1, bounds);
    LOG(layer.str());
    LOG(std::to_string(layer.len()));

};


void testSampling() {

    ActionSampling2D sm = ActionSampling2D("Test", 10);

    sm.str();

    bool keep = false;
    for (int i = 0; i < 28; i++) {
        sm.call(keep);
        if (!sm.is_done()) {
            sm.update(utils::random.get_random_float());
        };

        if (i == (sm.len() + 3)) {
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

    PCNN model = PCNN(3, Nj, 10., 0.1, 0.4, 0.1, 0.1,
                      0.5, 8, 0.1, xfilter, "2D");

    LOG(model.str());

    LOG("-- input 1");
    Eigen::Vector2f x = {0.2, 0.2};
    Eigen::VectorXf y = model.call(x);

    LOG("model length: " + std::to_string(model.len()));

    LOG("-- input 2");

    x = {0.1, 0.1};
    y = model.call(x);
    LOG("model length: " + std::to_string(model.len()));

    LOG("---");
    LOG("connectivity:");
    utils::logging.log_matrix(model.get_connectivity());

    LOG("wrec:");
    utils::logging.log_matrix(model.get_wrec());

    SPACE("#---#");
}

};
