#ifndef CONVLAYER_HPP
#define CONVLAYER_HPP

#pragma once
#include "Matrix.hpp"
#include "utils_cnn.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>
#include <cmath>


template <typename T>
class ConvLayer {
private:
    std::vector<std::vector<Matrix<T>>> filters_; // [filtro][canal]
    ConvLayeryperparameters h_;


    Matrix<T> applyPadding(const Matrix<T>& input, int padding) const {
        size_t new_rows = input.rows() + 2 * padding;
        size_t new_cols = input.cols() + 2 * padding;
        Matrix<T> padded(new_rows, new_cols, 0);

        for (size_t i = 0; i < input.rows(); ++i)
            for (size_t j = 0; j < input.cols(); ++j)
                padded[i + padding][j + padding] = input[i][j];

        return padded;
    }

    Matrix<T> relu(const Matrix<T>& m) const {
        Matrix<T> result = m;
        for (size_t i = 0; i < m.rows(); ++i)
            for (size_t j = 0; j < m.cols(); ++j)
                result[i][j] = std::max(static_cast<T>(0), m[i][j]);
        return result;
    }

    Matrix<T> pooling(const Matrix<T>& m) const {
        size_t out_rows = m.rows() / h_.pool_size;
        size_t out_cols = m.cols() / h_.pool_size;
        Matrix<T> output(out_rows, out_cols, 0);

        for (size_t i = 0; i < out_rows; ++i) {
            for (size_t j = 0; j < out_cols; ++j) {
                T val;
                if (h_.pool_mode == PoolMode::MAX) val = std::numeric_limits<T>::lowest();
                else if (h_.pool_mode == PoolMode::MIN) val = std::numeric_limits<T>::max();
                else val = 0;

                for (size_t pi = 0; pi < h_.pool_size; ++pi) {
                    for (size_t pj = 0; pj < h_.pool_size; ++pj) {
                        T current = m[i * h_.pool_size + pi][j * h_.pool_size + pj];

                        if (h_.pool_mode == PoolMode::MAX)
                            val = std::max(val, current);
                        else if (h_.pool_mode == PoolMode::MIN)
                            val = std::min(val, current);
                        else
                            val += current;
                    }
                }

                if (h_.pool_mode == PoolMode::AVG)
                    val /= (h_.pool_size * h_.pool_size);

                output[i][j] = val;
            }
        }
        return output;
    }

    Matrix<T> convolveMultiChannel(const std::vector<Matrix<T>>& input_channels,
        const std::vector<Matrix<T>>& kernel_set) const {
        if (input_channels.size() != kernel_set.size())
            throw std::invalid_argument("Cantidad de canales no coincide con kernels del filtro");

        Matrix<T> result;

        for (size_t c = 0; c < input_channels.size(); ++c) {

            std::cout << "++++++++++APLICANDO PADDING++++++++++\n\n";

            Matrix<T> padded = applyPadding(input_channels[c], h_.padding);

            if (h_.debug) std::cout << padded << '\n';

            const Matrix<T>& kernel = kernel_set[c];
            size_t k = kernel.rows();

            size_t out_rows = (padded.rows() - k) / h_.stride + 1;
            size_t out_cols = (padded.cols() - k) / h_.stride + 1;
            Matrix<T> conv(out_rows, out_cols, 0);

            for (size_t i = 0; i < out_rows; ++i) {
                for (size_t j = 0; j < out_cols; ++j) {
                    T acc = 0;
                    for (size_t ki = 0; ki < k; ++ki)
                        for (size_t kj = 0; kj < k; ++kj)
                            acc += kernel[ki][kj] * padded[i * h_.stride + ki][j * h_.stride + kj];
                    conv[i][j] = acc;
                }
            }

            result = (c == 0) ? conv : (result + conv);
        }

        return result;
    }

public:
    ConvLayer(const std::vector<std::vector<Matrix<T>>>& filters, const ConvLayeryperparameters& h)
        : filters_(filters), h_(h) {}

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input_channels) const {
        std::vector<Matrix<T>> outputs;
        size_t k = 0;

        for (const auto& kernel_set : filters_) {

            std::cout << "*************COLVOLUCION " << k << "*************\n";
            Matrix<T> feature = convolveMultiChannel(input_channels, kernel_set);
            if (h_.debug) std::cout << feature << "\n";
        
            if (h_.relu) {
                std::cout << "*************ACTIVACION*************\n\n";
                feature = relu(feature);
                if (h_.debug) std::cout << feature << "\n";
            }
        
            if (h_.pooling) {
                std::cout << "*************POOLING*************\n\n";
                feature = pooling(feature);
                if (h_.debug) std::cout << feature << "\n";
            }

            outputs.push_back(feature);

            ++k;
        }

        return outputs;
    }


    const std::vector<std::vector<Matrix<T>>>& getFilters() const { return filters_; }
    int getStride() const { return h_.stride; }
    int getPadding() const { return h_.padding; }
    bool usesPooling() const { return h_.pooling; }
    size_t getPoolSize() const { return h_.pool_size; }

};


#endif