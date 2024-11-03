#pragma once
#include <string>
#include <Eigen/Dense>

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

private:
    const int N;
    const double gain_;
    const double offset_;
    Eigen::Matrix2d W_;
    Eigen::Matrix2d C_;
    Eigen::Matrix2d mask_;
    Eigen::Vector2d u_;
};
