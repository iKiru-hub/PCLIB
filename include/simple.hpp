#pragma once
#include <string>
#include <Eigen/Dense>

class Simple {
public:
    Simple(int value);
    int getValue() const;
    void setValue(int value);
    double multiply(double factor);
    std::string toString() const;
    
    // Eigen-based methods with explicit specifications
    Eigen::VectorXd createVector(int size) const;
    Eigen::MatrixXd createMatrix(int rows, int cols) const;
    double computeVectorNorm(const Eigen::VectorXd& vec) const;
    Eigen::VectorXd multiplyVector(const Eigen::VectorXd& vec, double scalar) const;

private:
    int m_value;
};
