#pragma once
#include "Matrix.hpp"
#include <random>

enum class Initializer { U_XAVIER, N_XAVIER, HE, RANDOM };

enum class Optimizer { ADAM, RMS_PROP, NONE };

enum class Activation { SIGMOID, RELU, TANH };


double sigmoid(const double& _x) {
    return 1.0 / (1.0 + exp(-_x));
}

double sigmoid_deri(const double& _x) {
    const auto f = sigmoid(_x);
    return f * (1.0 - f);
}

double relu(const double& _x) {
    return std::min(0.0, _x);
}

double relu_deri(const double& _x) {
    return _x > 0.0 ? 1.0 : 0.0;
}

double tanhm(const double& _x) {
    return (1.0 - std::exp(-2 * _x)) / (1.0 + std::exp(-2 * _x));
}

double tanh_deri(const double& _x) {
    const auto f = tanhm(_x);
    return 1.0 - f * f;
}

Matrix<double> softmax(const Matrix<double>& _m) {
    const auto ROWS = _m.rows();

    Matrix result(ROWS, 1, 0.0);

    double max_val = _m[0][0];
    for (size_t i = 1; i < ROWS; ++i)
        if (_m[i][0] > max_val) max_val = _m[i][0];

    double sum_exp = 0.0;
    for (size_t i = 0; i < ROWS; ++i) {
        result[i][0] = std::exp(_m[i][0] - max_val);
        sum_exp += result[i][0];
    }

    for (size_t i = 0; i < ROWS; ++i)
        result[i][0] /= sum_exp;

    return result;
}

double cross_entropy_loss(const Matrix<double>& _y_predict, const Matrix<double>& _y_true) {
    double loss = 0.0;
    for (size_t i = 0; i < _y_predict.rows(); ++i)
        if (_y_true[i][0] == 1.0)
            loss = -std::log(_y_predict[i][0] + 1e-9);
    return loss;
}

Matrix<double> xavier_uniform_init(const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(6.0 / (_input_size + _output_size));
    std::uniform_real_distribution<> dist(-limit, limit);

    Matrix m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

Matrix<double> xavier_normal_init(const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(2.0 / (_input_size + _output_size));
    std::normal_distribution<> dist(0.0, limit);

    Matrix m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

Matrix<double> he_init(const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(2.0 / _input_size);
    std::normal_distribution<> dist(0.0, limit);

    Matrix m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

Matrix<double> random_init(const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    std::uniform_real_distribution<> dist(0.0, 1.0);

    Matrix m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

double square(const double& _x) {
    return _x * _x;
}


struct MLPHyperparameters {
    std::vector<size_t> layers = { 2, 2, 3 };
    Initializer initializer = Initializer::RANDOM;
    Optimizer optimizer = Optimizer::NONE;
    Activation activation = Activation::SIGMOID;
    double learning_rate = 0.01;
    size_t batch = 16;
    size_t epochs = 10;
    double decay_rate = 0.9;
    double epsilon = 1e-8;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double bias_init = 0.0;
    size_t timestep = 1;
    bool shuffle = false;
    bool debug = false;
};

