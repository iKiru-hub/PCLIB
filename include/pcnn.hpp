#include <iostream>
#include <Eigen/Dense>
#include "utils.hpp"
#include <unordered_map>
#include <memory>
#include <array>
#include <algorithm>
#include <cstdio>
#include <tuple>
#include <cassert>


/* #define LOG(msg) utils::logging.log(msg, "PCLIB") */
#define SPACE utils::logging.space

// blank log function
void LOG(const std::string& msg) {
    std::cout << msg << std::endl;
}

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



/* ========================================== */
/* ============ local functions ============= */
/* ========================================== */


// @brief boundary conditions for an hexagon
/*
INITIALIZATION:
- an hexagon of side 1
- center at C=(0, 0)
- nodes
- apothem length 0.86602540378

GIVEN:
- point P=(x, y)

PROCEDURE:

% checkpoint: short distance PO < apothem

% calculate the boundary conditions
1. reflect the point wrt the center C -> R
2. determine the two closest nodes A, B
3. calculate the intersection S between the line
   RO and the side of the hexagon AB

% checkpoint: no intersection S, point is inside the hexagon

4. reflect the point OP wrt the intersection S, this is the
   new point P
*/

/*
m = {
    {(-0.5+0.5)/2, (-0.86602540378-0.86602540378)/2},
    {(1+0.5)/2, (-0.86602540378+0)/2},
    {(1+0.5)/2, (0+0.86602540378)/2},
    {(0.5-0.5)/2, (0.86602540378+0.86602540378)/2},
    {(-0.5-1)/2, (0.86602540378+0)/2},
    {(-1-0.5)/2, (0-0.86602540378)/2}
}
for i=1,#m,1
do
    print(m[i][1].."; "..m[i][2])
end
*/

class Hexagon {

    // center: (0, 0)
    // side: 1
    std::array<std::array<float, 2>, 6> centers;
    std::array<size_t, 6> index = {0, 1, 2, 3, 4, 5};

    // @brief check whether p is within the inner circle
    bool apothem_checkpoint(float x, float y) {
        float dist = std::sqrt(std::pow(x, 2) + \
                               std::pow(y, 2));
        bool within = dist < 0.86602540378f;
        if (within) {
            return true;
        } else {
            return false;
        }
    }

    // @brief wrap the point to the boundary
    int wrap(float x, float y, float *p_new_x, float *p_new_y) {

        // reflect the point p to r wrt the center (0, 0)
        float rx = x * -1;
        float ry = y * -1;

        /* printf("Reflection: rx %f, ry %f\n", rx, ry); */

        // calculate and sort the distances to the centers
        std::array<float, 6> distances;
        for (int i = 0; i < 6; i++) {
            distances[i] = std::sqrt(std::pow(centers[i][0] - rx,
                                              2) + \
                                     std::pow(centers[i][1] - ry,
                                              2));
        }

        // Sort the index array based on the
        // values in the original array
        std::sort(index.begin(), index.end(),
            [&distances](const size_t& a, const size_t& b) {
                return distances[a] < distances[b];
            }
        );

        float ax = centers[index[0]][0];
        float ay = centers[index[0]][1];
        float bx = centers[index[1]][0];
        float by = centers[index[1]][1];
        float mx = (ax + bx) / 2.0f;
        float my = (ay + by) / 2.0f;

        //
        /* printf("A: %f, %f\n", ax, ay); */
        /* printf("B: %f, %f\n", bx, by); */
        // calculate the intersection s between ab and ro
        float sx, sy;
        if (utils::get_segments_intersection(
            ax, ay, bx, by, rx, ry,
            0.0f, 0.0f, &sx, &sy)) {
        } else {
            // checkpoint: no intersection,
            // point is inside the hexagon
            /* LOG("[+] no intersection"); */
            return 0;
        }

        // reflect the point r wrt the intersection s
        /* *p_new_x = 2 * sx - rx; */
        /* *p_new_y = 2 * sy - ry; */
        rx = 2 * sx - rx;
        ry = 2 * sy - ry;

        // reflect wrt the line s-center
        std::array<float, 2> z;
        if (sy > 0) {
             z = utils::reflect_point_over_segment(
                rx, ry, 0.0f, 0.0f, mx, my);
        } else {
            z = utils::reflect_point_over_segment(
                rx, ry, mx, my, 0.0f, 0.0f);
        }

        *p_new_x = z[0];
        *p_new_y = z[1];

        return 1;
    }

public:

    Hexagon() {
        centers[0] = {-0.5f, -0.86602540378f};
        centers[1] = {0.5f, -0.86602540378f};
        centers[2] = {1.0f, 0.0f};
        centers[3] = {0.5f, 0.86602540378f};
        centers[4] = {-0.5, 0.86602540378f};
        centers[5] = {-1.0f, 0.0f};

        LOG("[+] hexagon created");
    }

    ~Hexagon() { LOG("[-] hexagon destroyed"); }

    // @brief call: apply the boundary conditions
    std::array<float, 2> call(float x, float y) {

        float new_x, new_y;

        if (!apothem_checkpoint(x, y)) {
            if (wrap(x, y, &new_x, &new_y)) {
                /* LOG("[+] point wrapped to boundary"); */
                return {new_x, new_y};
            } else {
                /* LOG("[+] point within the hexagon"); */
                return {x, y};
            }
        } else {
            /* LOG("[+] within the apothem"); */
            return {x, y};
        }
    }

    std::string str() { return "hexagon"; }
    std::string repr() { return str(); }
    std::array<std::array<float, 2>, 6> get_centers() {
        return centers;
    }
};

struct GSparams {
    float offset;
    float gain;
    float clip;

    GSparams(float offset, float gain, float clip):
        offset(offset), gain(gain), clip(clip) {}
};


/* ========================================== */
/* ============== INPUT LAYER =============== */
/* ========================================== */

// @brief InputLayer abstract class
class InputLayer {

public:
    InputLayer(int N) {

        // assert N is a perfect square
        assert(std::sqrt(N) == std::floor(std::sqrt(N)));
        this->N = N;

        // Initialize the centers
        basis = Eigen::MatrixXf::Zero(N, 2);
        y = Eigen::VectorXf::Zero(N);
    }

    virtual ~InputLayer() { LOG("[-] InputLayer destroyed"); }
    virtual Eigen::VectorXf call(const Eigen::Vector2f& x){};

    std::string str() { return "InputLayer"; }
    std::string repr() { return str() + "(N=" + \
        std::to_string(N) + ")"; }
    Eigen::MatrixXf get_centers() { return basis; };
    Eigen::VectorXf get_activation() { return y; }

    int len() const { return N; }

private:
    virtual void make_tuning() {};

protected:
    int N;
    Eigen::MatrixXf basis;
    Eigen::VectorXf y;
};


class PCLayer : public InputLayer {

public:

    PCLayer(int n, float sigma,
            std::array<float, 4> bounds)
        : InputLayer(n*n), n(n), sigma(sigma),
        bounds(bounds) {

        // Compute the centers
        make_tuning();
        LOG("[+] PCLayer created");
    }

    ~PCLayer(){ LOG("[-] PCLayer destroyed"); }

    // @brief Call the PCLayer with a 2D input and compute
    // the Gaussian distance to the centers
    Eigen::VectorXf call(const Eigen::Vector2f& x) {
        for (int i = 0; i < N; i++) {
            float dx = x(0) - basis(i, 0);
            float dy = x(1) - basis(i, 1);
            float dist_squared = std::pow(dx, 2) + \
                std::pow(dy, 2);
            y(i) = std::exp(-dist_squared / denom);
        }
        return y;
    }

    std::string str() { return "PCLayer"; }

private:

    int n;
    float sigma;
    const float denom = 2 * sigma * sigma;
    std::array<float, 4> bounds;

    void make_tuning() {

        // calculate the spacing between the centers
        // given the number of centers and the bounds
        float x_spacing = (bounds[1] - bounds[0]) / (n - 1);
        float y_spacing = (bounds[3] - bounds[2]) / (n - 1);

        // Compute the centers
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                basis(i * n + j, 0) = bounds[0] + \
                    i * x_spacing;
                basis(i * n + j, 1) = bounds[2] + \
                    j * y_spacing;
            }
        }
    }

};


class RandLayer : public InputLayer {

public:

    RandLayer(int N): InputLayer(N) {

        // make matrix
        make_tuning();
        LOG("[+] RandLayer created");
    }

    ~RandLayer() { LOG("[-] PCLayer destroyed"); }

    // @brief Call the RandLayer with a 2D input and
    // apply a the layer's linear projection
    Eigen::VectorXf call(const Eigen::Vector2f& x) {
        for (int i = 0; i < N; i++) {
            y(i) = basis.row(i).dot(x);
        }
        return y;
    }

    std::string str() { return "RandLayer"; }

private:

    // @brief calculate the basis of a linear projection
    // 2D -> nD as set of orthogonal vectors through
    // the Gram-Schmidt process
    void make_tuning() {

        /* size_t n = N; */
        /* std::array<std::array<float, n>, 2> matrix = utils::make_orthonormal_matrix<2, n>(); */

        // define two random vectors of length N
        // in the range [0, 1]
        Eigen::VectorXf v1 = (Eigen::VectorXf::Random(N) + \
            Eigen::VectorXf::Ones(N)) / 2;
        Eigen::VectorXf v2 = (Eigen::VectorXf::Random(N) + \
            Eigen::VectorXf::Ones(N)) / 2;

        // Compute dot products
        float multiplier = v1.dot(v2) / v1.dot(v1);
        Eigen::VectorXf v2_orth = v2 - multiplier * v1;

        // compute the sum of v1
        float sum = v1.sum();
        float sum_orth = v2_orth.sum();

        // normalize the vectors
        v1 = v1 / sum;
        v2_orth = v2_orth / sum_orth;

        // define final matrix
        basis.col(0) = v1;
        basis.col(1) = v2_orth;
    }

};


enum BoundaryType {
    square,
    hexagon,
    circle,
    klein
};

class GridLayer : public InputLayer {

public:

    GridLayer(int N, float sigma, float speed,
              std::array<float, 4> init_bounds = {-1, 1, -1, 1},
              std::string boundary_type = "square",
              std::string basis_type = "square"):
        InputLayer(N), sigma(sigma), speed(speed),
        boundary_type(boundary_type),
        basis_type(basis_type),
        init_bounds(init_bounds){

        this->basis = Eigen::MatrixXf::Zero(N, 2);
        this->positions = Eigen::MatrixXf::Zero(N, 2);

        // record positions
        if (basis_type == "square") {
            square_basis();
        } else if (basis_type == "random_square") {
            random_square_basis();
        } else if (basis_type == "shifted_square") {
            shifted_square_basis();
        } else if (basis_type == "random_circle") {
            random_circle_basis();
        } else {
            throw std::invalid_argument(
                "CUSTOM ERROR: unknown basis type" );
        }

        // record initial positions in the basis
        // pass by value
        positions = basis;

        // record boundary type
        if (boundary_type == "square") {
            boundary_type_num = BoundaryType::square;
        } else if (boundary_type == "hexagon") {
            boundary_type_num = BoundaryType::hexagon;
        } else if (boundary_type == "circle") {
            boundary_type_num = BoundaryType::circle;
        } else if (boundary_type == "klein") {
            boundary_type_num = BoundaryType::klein;
        } else {
            throw std::invalid_argument(
                "CUSTOM ERROR: unknown boundary type" );
        }

        DEBUG("Boundary type: " + boundary_type);
        DEBUG("Basis type: " + basis_type);

        // make matrix
        LOG("[+] GridLayer created");
    }

    ~GridLayer() { LOG("[-] GridLayer destroyed"); }

    // @brief call the GridLayer with a 2D input
    Eigen::VectorXf call(const std::array<float, 2>& v) {

        // update position with velociy
        for (int i = 0; i < N; i++) {
            positions(i, 0) += speed * v[0];
            positions(i, 1) += speed * v[1];
        }

        // apply boundary conditions
        boundary_conditions();

        // compute the activation
        calc_activation();
        return y;
    }

    Eigen::VectorXf fwd_position(
        const std::array<float, 2>& v) {

        /* Eigen::MatrixXf new_positions = Eigen::MatrixXf::Zero(N, 2); */
        /* std::array<std::array<float, 2>, 25> new_positions; */
        Eigen::MatrixXf new_positions;
        /* new_positions.col(0) = positions.col(0) + speed * \ */
            /* v[0]; */
        /* new_positions.col(1) = positions.col(1) + speed * \ */
            /* v[1]; */
        for (int i = 0; i < N; i++) {
            new_positions(i, 0) = positions(i, 0) + \
                speed * (v[0] - positions(i, 0));
            new_positions(i, 1) = positions(i, 1) + \
                speed * (v[1] - positions(i, 1));
        }

        // check boundary conditions
        /* for (int i = 0; i < N; i++) { */
        /*     std::array<float, 2> new_position = hexagon.call( */
        /*         new_positions[i][0], new_positions[i][1]); */

        /*     new_positions[i][0] = new_position[0]; */
        /*     new_positions[i][1] = new_position[1]; */
            /* std::array<float, 2> new_position = hexagon.call( */
            /*     new_positions(i, 0), new_positions(i, 1)); */
            /* new_positions(i, 0) = new_position[0]; */
            /* new_positions(i, 1) = new_position[1]; */
        /* } */

        boundary_conditions(new_positions);

        // compute the activation
        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        float dist_squared;
        for (int i = 0; i < N; i++) {
            /* dist_squared = std::pow(new_positions(i, 0), 2) + \ */
            /*     std::pow(new_positions(i, 1), 2); */
            dist_squared = std::pow(new_positions(i, 0),
                                    2) + \
                std::pow(new_positions(i, 1), 2);
            yfwd(i) = std::exp(-dist_squared / sigma);
        }

        return yfwd;
    }

    std::string str() const { return "GridLayer"; }
    std::string repr() const { return "GridLayer(" + \
        boundary_type + ", " + basis_type + ")"; }
    Eigen::MatrixXf get_positions() { return positions; }
    void reset(std::array<float, 2> v) {
        this->positions = basis;
        call(v);
    }

private:
    Eigen::MatrixXf positions;
    float sigma;
    float speed;
    int boundary_type_num;
    std::string boundary_type;
    std::string basis_type;
    Hexagon hexagon;
    std::array<float, 4> init_bounds;

    // define basis type
    void square_basis() {

        int n = static_cast<int>(std::sqrt(N));
        if (n*n != N) {
            LOG("WARNING: downsizing to " + \
                std::to_string(n*n));
            this-> N = n*n;
        }

        float dx = 1.0f / (static_cast<float>(n) + 0.0f);

        // define the centers as a grid excluding
        /* Eigen::VectorXf y = utils::linspace( */
        /*                 0.0f, 1.0f-dx, n); */
        /* Eigen::VectorXf x = utils::linspace( */
        /*                 0.0f, 1.0f-dx, n); */
        Eigen::VectorXf y = utils::linspace(
                        init_bounds[0],
                        init_bounds[1]-dx, n);
        Eigen::VectorXf x = utils::linspace(
                        init_bounds[2],
                        init_bounds[3]-dx, n);

        for (std::size_t i=0; i<N; i++) {
            float xi = x(i / n);
            float yi = y(i % n);
            basis(i, 0) = xi;
            basis(i, 1) = yi;
        }
    }

    void random_square_basis() {

        // sample random points within the unit circle
        for (int i = 0; i < N; i++) {
            float x = utils::random.get_random_float(0, 1);
            float y = utils::random.get_random_float(0, 1);
            basis(i, 0) = x;
            basis(i, 1) = y;
        }
    }

    void shifted_square_basis() {

        int n = static_cast<int>(std::sqrt(N));
        if (n * n != N) {
            LOG("WARNING: downsizing to " + \
                std::to_string(n * n));
            this->N = n * n;
        }
        // Side length of hexagon
        float s = 1.0f / static_cast<float>(n);
        // Horizontal spacing
        float dx = std::sqrt(3.0f) / 2.0f * s;

        // Vertical spacing
        float dy = 1.5f * s;

        // Grid positions
        Eigen::MatrixXf basis(N, 2);

        // Iterate over rows
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int idx = i * n + j;
                basis(idx, 0) = j * dx;
                basis(idx, 1) = i * dy + (j % 2) * (s / 2);
            }
        }

        // Normalize positions to fit within [0, 1] range
        basis.col(0).array() /= (n * dx);
        basis.col(1).array() /= (n * dy);

        this->basis = basis;
    }

    void random_circle_basis() {

        // sample random points within the unit circle
        for (int i = 0; i < N; i++) {
            float theta = utils::random.get_random_float(
                                        0.0f, 6.28f);
            float radius = utils::random.get_random_float(0, 1);
            basis(i, 0) = radius * std::cos(theta);
            basis(i, 1) = radius * std::sin(theta);
        }
    }

    // define boundary type
    void boundary_conditions(Eigen::MatrixXf& pos) {
        for (int i = 0; i < N; i++) {
            std::array<float, 2> new_position = \
                apply_boundary(pos(i, 0), pos(i, 1));
            pos(i, 0) = new_position[0];
            pos(i, 1) = new_position[1];
        }
    }
    // define boundary type
    void boundary_conditions() {
        for (int i = 0; i < N; i++) {
            std::array<float, 2> new_position = apply_boundary(
                            positions(i, 0),
                            positions(i, 1));
            positions(i, 0) = new_position[0];
            positions(i, 1) = new_position[1];
        }
    }

    // gaussian distance of each position to the centers
    void calc_activation() {
        float dist_squared;
        for (int i = 0; i < N; i++) {

            /* dist_squared = std::pow( */
            /*     positions(i, 0) - 0.5f, 2) + \ */
            /*     std::pow(positions(i, 1) - 0.5f, 2); */
            dist_squared = std::pow(positions(i, 0), 2) + \
                std::pow(positions(i, 1), 2);

            y(i) = std::exp(-dist_squared / sigma);
        }
    }

    // apply boundary conditions
    std::array<float, 2> apply_boundary(float x, float y) {

        switch (boundary_type_num) {
            case BoundaryType::square:
                if (x < -1.0) { x += 2.0; }
                else if (x > 1.0) { x -= 2.0; }
                if (y < -1.0) { y += 2.0; }
                else if (y > 1.0) { y -= 2.0; }
                break;
            case BoundaryType::klein:
                if (x < -1.0) { x += 2.0; y = 2.0 - y; }
                else if (x > 1.0) { x -= 2.0; y = 2.0 - y; }
                if (y < -1.0) { y += 2.0; x = 2.0 - x;}
                else if (y > 1.0) { y -= 2.0; x = 2.0 - x;}
                break;
            case BoundaryType::circle:
                utils::logging.log("not implemented yet");
                break;
            case BoundaryType::hexagon:
                std::array<float, 2> P = hexagon.call(x, y);
                x = P[0];
                y = P[1];

        /* positions(i, 0) = x; */
        /* positions(i, 1) = y; */
        }

        return {x, y};
    };

};


class GridNetwork {

public:

    GridNetwork(std::vector<GridLayer> layers)
        : layers(layers) {

        // Initialize the variables
        this->num_layers = layers.size();

        int total_size = 0;
        full_repr = "(";
        for (auto& layer : layers) {
                total_size += layer.len();
                full_repr += layer.repr();
            }
        full_repr += ")";
        this->N = total_size;
        y = Eigen::VectorXf::Zero(N);
        basis = Eigen::MatrixXf::Zero(total_size, 2);

        LOG("[+] GridNetwork created");
    }

    ~GridNetwork() { LOG("[-] GridNetwork destroyed"); }

    Eigen::VectorXf call(const std::array<float, 2>& x) {

        for (int i = 0; i < num_layers; i++) {
            y.segment(i*layers[i].len(), layers[i].len()) = \
                layers[i].call(x);
        }

        return y;
    }

    Eigen::VectorXf fwd_position(
        const std::array<float, 2>& v) {

        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < num_layers; i++) {
            yfwd.segment(i*layers[i].len(), layers[i].len()) = \
                layers[i].fwd_position(v);
        }

        return yfwd;
    }

    int len() const { return N; }
    int get_num_layers() const { return num_layers; }
    std::string str() const { return "GridNetwork"; }
    std::string repr() const {
        return str() + "(" + full_repr + ", N=" + \
        std::to_string(N) + ")"; }
    Eigen::VectorXf get_activation() const { return y; }
    Eigen::MatrixXf get_centers() {
        for (int i = 0; i < num_layers; i++) {
            basis.block(i*layers[i].len(),
                        0, layers[i].len(), 2) = \
                layers[i].get_positions();
        }
        return basis;
    }
    Eigen::MatrixXf get_positions() {
        for (int i = 0; i < num_layers; i++) {
            basis.block(i*layers[i].len(),
                        0, layers[i].len(), 2) = \
                layers[i].get_positions();
        }
        return basis;
    }

    void reset(std::array<float, 2> v) {
        for (int i = 0; i < num_layers; i++) {
            layers[i].reset(v);
        }
    }

private:
    std::vector<GridLayer> layers;
    int N;
    int num_layers;
    std::string full_repr;
    Eigen::VectorXf y;
    Eigen::MatrixXf basis;
};

// === purely hexagonal grid network ===

class GridHexLayer {

public:

    GridHexLayer(float sigma, float speed,
                 float offset_dx = 0.0f,
                 float offset_dy = 0.0f):
        sigma(sigma), speed(speed), hexagon(Hexagon()){

        // make matrix
        LOG("[+] GridHexLayer created");

        // apply the offset by stepping
        if (offset_dx != 0.0f && offset_dy != 0.0f) {
            call({offset_dx, offset_dy});
        }
    }

    ~GridHexLayer() { LOG("[-] GridHexLayer destroyed"); }

    // @brief call the GridLayer with a 2D input
    Eigen::VectorXf \
    call(const std::array<float, 2>& v) {

        // update position with velociy
        for (int i = 0; i < N; i++) {
            positions[i][0] = positions[i][0] + speed * v[0];
            positions[i][1] = positions[i][1] + speed * v[1];
        }

        // apply boundary conditions
        boundary_conditions();

        // compute the activation
        calc_activation();
        return y;
    }

    Eigen::VectorXf fwd_position(
        const std::array<float, 2>& v) {

        /* Eigen::MatrixXf new_positions = Eigen::MatrixXf::Zero(N, 2); */
        std::array<std::array<float, 2>, 25> new_positions;
        /* new_positions.col(0) = positions.col(0) + speed * \ */
            /* v[0]; */
        /* new_positions.col(1) = positions.col(1) + speed * \ */
            /* v[1]; */
        for (int i = 0; i < N; i++) {
            new_positions[i][0] = positions[i][0] + \
                speed * (v[0] - positions[i][0]);
            new_positions[i][1] = positions[i][1] + \
                speed * (v[1] - positions[i][1]);
        }

        // check boundary conditions
        for (int i = 0; i < N; i++) {
            std::array<float, 2> new_position = hexagon.call(
                new_positions[i][0], new_positions[i][1]);
            new_positions[i][0] = new_position[0];
            new_positions[i][1] = new_position[1];
            /* std::array<float, 2> new_position = hexagon.call( */
            /*     new_positions(i, 0), new_positions(i, 1)); */
            /* new_positions(i, 0) = new_position[0]; */
            /* new_positions(i, 1) = new_position[1]; */
        }

        // compute the activation
        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        float dist_squared;
        for (int i = 0; i < N; i++) {
            /* dist_squared = std::pow(new_positions(i, 0), 2) + \ */
            /*     std::pow(new_positions(i, 1), 2); */
            dist_squared = std::pow(new_positions[i][0], 2) + \
                std::pow(new_positions[i][1], 2);
            yfwd(i) = std::exp(-dist_squared / sigma);
        }

        return yfwd;
    }

    int len() const { return N; }
    std::string str() const { return "GridHexLayer"; }
    std::string repr() const { return "GridHexLayer"; }
    std::array<std::array<float, 2>, 25> get_positions()
    { return positions; }
    std::array<std::array<float, 2>, 25> get_centers()
    { return basis; }
    void reset(std::array<float, 2> v) {
        this->positions = basis;
        call(v);
    }


private:
    const int N = 25;


    const std::array<std::array<float, 2>, 25> basis = {{
        {0.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, std::sin((float)M_PI/3.0f)},
        {0.5f, std::sin((float)M_PI/3.0f)},
        {1.0f, 0.0f},
        {0.5f, -std::sin((float)M_PI/3.0f)},
        {-0.5f, -std::sin((float)M_PI/3.0f)},
        {-1.0f, 0.0f},
        {0.0f, 4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, -4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, 5.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, 4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, -5.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, -4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, 0.0f}
    }};
    std::array<float, 4> init_bounds = {-1.0f, 1.0f, -1.0f, 1.0f};
    std::array<std::array<float, 2>, 25> positions = {{
        {0.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, std::sin((float)M_PI/3.0f)},
        {0.5f, std::sin((float)M_PI/3.0f)},
        {1.0f, 0.0f},
        {0.5f, -std::sin((float)M_PI/3.0f)},
        {-0.5f, -std::sin((float)M_PI/3.0f)},
        {-1.0f, 0.0f},
        {0.0f, 4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, -4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, 5.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, 4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, -5.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, -4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, 0.0f}
    }};

    /* std::array<float, 7> y; */
    Eigen::VectorXf y = Eigen::VectorXf::Zero(25);
    float sigma;
    float speed;
    Hexagon hexagon;

    // define boundary type
    void boundary_conditions() {
        for (int i = 0; i < N; i++) {
            std::array<float, 2> new_position = hexagon.call(
                positions[i][0], positions[i][1]);
            this->positions[i][0] = new_position[0];
            this->positions[i][1] = new_position[1];
        }
    }

    // gaussian distance of each position to the centers
    void calc_activation() {
        float dist_squared;
        for (int i = 0; i < N; i++) {
            dist_squared = std::pow(positions[i][0], 2) + \
                std::pow(positions[i][1], 2);
            y(i) = std::exp(-dist_squared / sigma);
        }
    }

};


class GridHexNetwork {

public:

    GridHexNetwork(std::vector<GridHexLayer> layers)
        : layers(layers) {

        // Initialize the variables
        this->num_layers = layers.size();

        int total_size = 0;
        full_repr = "(";
        for (auto& layer : layers) {
                total_size += layer.len();
                full_repr += layer.repr();
            }
        full_repr += ")";
        this->N = total_size;
        y = Eigen::VectorXf::Zero(N);
        basis = Eigen::MatrixXf::Zero(total_size, 2);
        positions = Eigen::MatrixXf::Zero(total_size, 2);

        LOG("[+] GridHexNetwork created");
    }

    ~GridHexNetwork() { LOG("[-] GridHexNetwork destroyed"); }

    Eigen::VectorXf call(const std::array<float, 2>& x) {
        for (int i = 0; i < num_layers; i++) {
            y.segment(i*layers[i].len(), layers[i].len()) = \
                layers[i].call(x);
        }

        return y;
    }

    Eigen::VectorXf fwd_position(
        const std::array<float, 2>& x) {

        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < num_layers; i++) {
            yfwd.segment(i*layers[i].len(), layers[i].len()) = \
                layers[i].fwd_position(x);
        }

        return yfwd;
    }

    int len() const { return N; }
    int get_num_layers() const { return num_layers; }
    std::string str() const { return "GridNetwork"; }
    std::string repr() const {
        return str() + "(" + full_repr + ", N=" + \
        std::to_string(N) + ")"; }
    Eigen::VectorXf get_activation() const { return y; }
    Eigen::MatrixXf get_centers() {
        for (int i = 0; i < num_layers; i++) {
            int layer_len = layers[i].len();
            std::array<std::array<float, 2>, 25> layer_basis = \
                layers[i].get_centers();

            for (int j = 1; j < (layer_len+1); j++) {
                 basis(i*layer_len+j, 0) = layer_basis[j][0];
                 basis(i*layer_len+j, 1) = layer_basis[j][1];
                 }
        }
        return basis;
    }

    Eigen::MatrixXf get_positions() {
        for (int i = 0; i < num_layers; i++) {
            int layer_len = layers[i].len();
            std::array<std::array<float, 2>, 25> layer_positions = \
                layers[i].get_positions();

            for (int j = 1; j < (layer_len+1); j++) {
                 positions(i*layer_len+j, 0) = layer_positions[j][0];
                 positions(i*layer_len+j, 1) = layer_positions[j][1];
                 }
        }
        return basis;
    }

    void reset(std::array<float, 2> v) {
        for (int i = 0; i < num_layers; i++) {
            layers[i].reset(v);
        }
    }

private:
    std::vector<GridHexLayer> layers;
    int N;
    int num_layers;
    std::string full_repr;
    Eigen::VectorXf y;
    Eigen::MatrixXf basis;
    Eigen::MatrixXf positions;
};


/* ========================================== */
/* ================= PCNN =================== */
/* ========================================== */


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
        xfilter(std::move(xfilter)),
        name(name) {

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

    ~PCNN() { LOG("[-] PCNN destroyed"); }

    Eigen::VectorXf call(const Eigen::Vector2f& x,
                         const bool frozen = false,
                         const bool traced = true) {

        // pass the input through the filter layer
        x_filtered = xfilter.call(x);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = Wff * x_filtered + pre_x;
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        /* u = utils::gaussian_distance(x_filtered, Wff, sigma); */

        u = utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);

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


class PCNNrand {
public:
    PCNNrand(int N, int Nj, float gain, float offset,
         float clip_min, float threshold,
         float rep_threshold,
         float rec_threshold,
         int num_neighbors, float trace_tau,
         RandLayer xfilter, std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors),
        trace_tau(trace_tau),
        xfilter(std::move(xfilter)),
        name(name) {

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

    ~ PCNNrand() { LOG("[-] PCNNrand destroyed"); }

    Eigen::VectorXf call(const Eigen::Vector2f& x,
                         const bool frozen = false,
                         const bool traced = true) {

        // pass the input through the filter layer
        x_filtered = xfilter.call(x);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = Wff * x_filtered + pre_x;
        /* u = utils::cosine_similarity_vector_matrix( */
        /*         x_filtered, Wff); */

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        u = utils::gaussian_distance(x_filtered, Wff, sigma);
        u = utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);

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

    RandLayer xfilter;

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


class PCNNgrid {
public:
    PCNNgrid(int N, int Nj, float gain, float offset,
         float clip_min, float threshold,
         float rep_threshold,
         float rec_threshold,
         int num_neighbors, float trace_tau,
         GridNetwork xfilter, std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors),
        trace_tau(trace_tau),
        xfilter(std::move(xfilter)),
        name(name) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        connectivity = Eigen::MatrixXf::Zero(N, N);
        centers = Eigen::MatrixXf::Zero(N, 2);
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

    ~PCNNgrid() { LOG("[-] PCNN destroyed"); }

    std::pair<Eigen::VectorXf,
    Eigen::VectorXf> call(const std::array<float, 2>& v,
                          const bool frozen = false,
                          const bool traced = true) {

        // pass the input through the filter layer
        x_filtered = xfilter.call(v);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = Wff * x_filtered + pre_x;
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(),
                                                          0.01);

        // maybe use cosine similarity?
        /* u = utils::gaussian_distance(x_filtered, Wff, sigma); */

        u = utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);

        // update the trace
        if (traced) {
            trace = (1 - trace_tau) * trace + trace_tau * u;
        }

        // update model
        /* if (!frozen) { */
        /*     update(x_filtered); */
        /* } */

        /* return u; */
        return std::make_pair(u, x_filtered);
    }

    // @brief update the model
    void update(float x = -1.0, float y = -1.0) {

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

        // trim the weight update
        /* delta_wff = (dw.array() > 0.01).select(dw, 0.0f); */

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

            // record new center
            centers.row(idx) = Eigen::Vector2f(x, y);
        }

    }

   /* Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) { */
   /*      std::pair<Eigen::VectorXf, Eigen::VectorXf> ans = call(x); */
   /*      return ans.first; */
   /*  } */
   Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) {

        // pass the input through the filter layer
        x_filtered = xfilter.fwd_position(x);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = \
            Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        return utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);
    }

   Eigen::VectorXf fwd_int(const Eigen::VectorXf& a) {
        return Wrec * a + pre_x;
    }

   /* Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) { */
   /*      return call(x, true, false); */
   /*  } */

   /* Eigen::VectorXf fwd_int(const Eigen::VectorXf& a) { */
        /* return Wrec * a + pre_x; */
    /* } */

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
    Eigen::VectorXf get_activation() const { return u; }
    Eigen::VectorXf get_activation_gcn() const {
        return xfilter.get_activation(); }
    Eigen::MatrixXf get_wff() const { return Wff; }
    Eigen::MatrixXf get_wrec() const { return Wrec; }
    Eigen::MatrixXf get_connectivity() const { return connectivity; }
    Eigen::MatrixXf get_centers() const { return centers; }
    Eigen::VectorXf get_trace() const { return trace; }
    float get_delta_update() const { return delta_wff; }
    Eigen::MatrixXf get_positions_gcn() {
        return xfilter.get_positions();
    }
    void reset_gcn(std::array<float, 2> v) {
        xfilter.reset(v);
    }

    /* Eigen::MatrixXf get_centers() { */
    /*     return utils::calc_centers_from_layer( */
    /*         Wff, xfilter.get_centers()); */
    /* } */

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

    GridNetwork xfilter;

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
    Eigen::MatrixXf centers;

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


class PCNNgridhex {
public:
    PCNNgridhex(int N, int Nj, float gain, float offset,
         float clip_min, float threshold,
         float rep_threshold,
         float rec_threshold,
         int num_neighbors, float trace_tau,
         GridHexNetwork xfilter, std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors),
        trace_tau(trace_tau),
        xfilter(std::move(xfilter)),
        name(name) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        connectivity = Eigen::MatrixXf::Zero(N, N);
        centers = Eigen::MatrixXf::Zero(N, 2);
        mask = Eigen::VectorXf::Zero(N);
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        delta_wff = 0.0;
        x_filtered = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
        cell_count = 0;

        ach = 1.0f;

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            free_indexes.push_back(i);
        }
        fixed_indexes = {};

        LOG("[+] PCNNgridhex created");
    }

    ~PCNNgridhex() { LOG("[-] PCNNgridhex destroyed"); }

    std::pair<Eigen::VectorXf,
    Eigen::VectorXf> call(const std::array<float, 2>& v,
                          const bool frozen = false,
                          const bool traced = false) {

        // pass the input through the filter layer
        x_filtered = xfilter.call(v);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        /* u = Wff * x_filtered + pre_x; */
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = \
            Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        /* u = utils::gaussian_distance(x_filtered, Wff, sigma); */

        u = utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);

        // update the trace
        if (traced) {
            trace = (1 - trace_tau) * trace + trace_tau * u;
        }

        // update model
        /* if (!frozen) { */
        /*     update(x_filtered); */
        /* } */

        /* return u; */
        return std::make_pair(u, x_filtered);
    }

    // @brief update the model
    void update(float x = -1.0, float y = -1.0) {

        make_indexes();

        // exit: a fixed neuron is above threshold
        if (check_fixed_indexes() != -1) {
            DEBUG("!Fixed index above threshold");
            /* printf("(-)Fixed index above threshold\n"); */
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
        Eigen::VectorXf dw = x_filtered - \
            Wff.row(idx).transpose();

        // trim the weight update
        /* delta_wff = (dw.array() > 0.01).select(dw, 0.0f); */

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
                /* printf("(-)Repulsion [%f]{%f}\n", similarity, */
                       /* rep_threshold); */
                return void();
            }

            // update count and backup
            cell_count++;
            Wffbackup.row(idx) = Wff.row(idx);
            /* printf("(:)cell_count: %d [%f]\n", cell_count, */
                   /* similarity); */

            // update recurrent connections
            update_recurrent();

            // record new center
            centers.row(idx) = Eigen::Vector2f(x, y);
        }

    }

   /* void fwd_ext(const std::array<float, 2>& x) {} */
   /* /1* Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) { *1/ */
   /* /1*      return call(x, true, false); *1/ */
   /* /1*  } *1/ */
   /* Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) { */
   /*      std::pair<Eigen::VectorXf, Eigen::VectorXf> ans = call(x); */
   Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) {

        // pass the input through the filter layer
        x_filtered = xfilter.fwd_position(x);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = \
            Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        return utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);
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
    std::string str() const { return "PCNNgridhex." + name; }
    std::string repr() const {
        return "PCNNgridhex(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
            std::to_string(rep_threshold) + \
            std::to_string(rec_threshold) + \
            std::to_string(num_neighbors) + \
            std::to_string(trace_tau) + ")";
    }
    Eigen::VectorXf get_activation() const { return u; }
    Eigen::VectorXf get_activation_gcn() const {
        return xfilter.get_activation(); }
    Eigen::MatrixXf get_wff() const { return Wff; }
    Eigen::MatrixXf get_wrec() const { return Wrec; }
    Eigen::MatrixXf get_connectivity() const { return connectivity; }
    Eigen::MatrixXf get_centers() const { return centers; }
    Eigen::VectorXf get_trace() const { return trace; }
    float get_delta_update() const { return delta_wff; }
    Eigen::MatrixXf get_positions_gcn() {
        return xfilter.get_positions();
    }
    void reset_gcn(std::array<float, 2> v) {
        xfilter.reset(v);
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

    GridHexNetwork xfilter;

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
    Eigen::MatrixXf centers;

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
            /* printf("(-)Fixed index above threshold [u=%f]{%f}\n", */
                   /* max_u, threshold); */
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


/* ========================================== */
/* =========== MODULATION MODULES =========== */
/* ========================================== */

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

        /* if (v < min_v) { */
        /*     v = min_v; */
        /* } */
        if (v < min_v) {
            v = 0.0f;
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
    int len() const { return 1; }
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


// base modulation | Dopamine & Boundary
class BaseModulation{
    float output;
    std::string name;
    int size;
    float lr;
    float threshold;
    std::vector<float> weights;
    LeakyVariable1D leaky;
    GSparams gsparams;

public:

    BaseModulation(std::string name, int size, float lr,
             float threshold, float offset, float gain,
             float clip, float eq, float tau,
             float min_v = 0.0f):
        name(name), size(size), weights(size, 0.0f),
        threshold(threshold), gsparams(offset, gain, clip),
        lr(lr), leaky(LeakyVariable1D(name, eq, tau, min_v)) {}

    ~BaseModulation() {}

    float call(const std::vector<float>& u,
               float x = 0.0f, bool simulate = false) {

        // forward to the leaky variable
        float v = leaky.call(x, simulate);

        // update the weights
        if (!simulate) {
            for (int i = 0; i < size; i++) {
                float ui = u[i] > threshold ? u[i] : 0.0;
                weights[i] += lr * v * ui;

                // clip the weights in (0, 1)
                if (weights[i] < 0.0) {
                    weights[i] = 0.0;
                } else if (weights[i] > 1.0) {
                    weights[i] = 1.0;
                }
            }
        }

        // compute the output
        output = 0.0;
        for (int i = 0; i < size; i++) {
            output += weights[i] * u[i];
        }

        // apply activation function
        output = utils::generalized_sigmoid(output,
                                            gsparams.offset,
                                            gsparams.gain,
                                            gsparams.clip);

        return output;
    }

    float get_output() { return output; }
    std::vector<float> get_weights() { return weights; }
    float get_leaky_v() { return leaky.get_v(); }
    std::string str() { return name; }
    std::string repr() { return name + "(1D)"; }
    int len() { return size; }
};

// program 1 : maximum activity
class PopulationMaxProgram {

    float output;

public:

    PopulationMaxProgram() {}
    ~PopulationMaxProgram() {}

    float call(const std::vector<float>& u) {
        output = *std::max_element(u.begin(), u.end());
        return output;
    }

    float get_value() { return output; }
    std::string str() { return "PopulationMaxProgram"; }
    std::string repr() { return "PopulationMaxProgram"; }
    int len() { return 1; }
};



class Circuits {

    BaseModulation& da;
    BaseModulation& bnd;
    PopulationMaxProgram pmax;

    std::array<float, 3> output;

public:

    Circuits(BaseModulation& da, BaseModulation& bnd):
        da(da), bnd(bnd), pmax(PopulationMaxProgram()) {}

    ~Circuits() {}

    std::array<float, 3> call(const std::vector<float>& u,
                                         float collision,
                                         float reward,
                                         bool simulate = false) {

        output[0] = da.call(u, reward, simulate);
        output[1] = bnd.call(u, collision, simulate);
        output[2] = pmax.call(u);

        return output;
    }

    std::string str() { return "Circuits"; }
    std::string repr() { return "Circuits"; }
    int len() { return 3; }
    std::array<float, 3> get_output() { return output; }
    std::array<float, 2> get_leaky_v() {
        return {da.get_leaky_v(), bnd.get_leaky_v()}; }
};


/* ========================================== */
/* ========= ACTION SAMPLING MODULE ========= */
/* ========================================== */


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
    int len() const { return num_samples; }
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


/* ========================================== */
/* =========== EXPERIENCE MODULE ============ */
/* ========================================== */


class ExperienceModule {

};



/* ========================================== */
/* ========================================== */


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
    /* RandLayer xfilter = RandLayer(Nj); */
    LOG(xfilter.str());

    PCNN model = PCNN(3, Nj, 10., 0.1, 0.4, 0.1, 0.1,
                      0.5, 8, 0.1, xfilter, "2D");

    LOG(model.str());

    LOG("-- input 1");
    Eigen::Vector2f x = {0.2, 0.2};
    Eigen::VectorXf y = model.call(x);
    model.update();
    LOG("model length: " + std::to_string(model.len()));

    LOG("-- input 2");

    x = {0.1, 0.1};
    y = model.call(x);
    model.update();
    LOG("model length: " + std::to_string(model.len()));

    LOG("---");
    LOG("connectivity:");
    utils::logging.log_matrix(model.get_connectivity());

    LOG("wrec:");
    utils::logging.log_matrix(model.get_wrec());

    SPACE("#---#");
}


void test_randlayer() {

    SPACE("#---#");

    LOG("Testing RandLayer...");

    int N = 5;

    RandLayer layer = RandLayer(N);
    LOG(layer.str());

    Eigen::Vector2f x = {0.2, 0.2};
    Eigen::VectorXf y = layer.call(x);

    // log y
    utils::logging.log_vector(y);

    LOG("layer length: " + std::to_string(layer.len()));

    LOG("matrix:");
    utils::logging.log_matrix(layer.get_centers());

    SPACE("#---#");
}


void test_gridlayer() {

    GridLayer gc(9, 0.1, 0.2);
    /* printf(gc.str()); */

    utils::logging.log_matrix(gc.get_positions());

    std::array<float, 2> x = {0.2, 0.2};
    gc.call(x);
}
};
