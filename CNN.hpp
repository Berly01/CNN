#ifndef CNN_HPP
#define CNN_HPP

#pragma once
#include "ConvLayer.hpp"
#include "MLP.hpp"

template <typename T>
class CNN {
private:
    std::vector<ConvLayer<T>> layers_;
    MLP mlp_;

public:
    CNN(const std::vector<ConvLayer<T>>& layers, const MLP& mlp) : layers_(layers), mlp_(mlp) {}

    Matrix<T> forward(const std::vector<Matrix<T>>& input) {
        std::vector<Matrix<T>> current = input;
        std::vector<T> flatten_output;

        for (const auto& layer : layers_) 
            current = layer.forward(current);
        
        for (const auto& fmap : current) {
            const auto flat = fmap.flat(); 
            flatten_output.insert(flatten_output.end(), flat.cbegin(), flat.cend());
        }

        auto feature = Matrix<double>(flatten_output);

        std::cout << "*******************MAPA DE CARACTERISTICAS*******************\n"
            << feature << '\n';

        auto p = mlp_.predict(feature);
        std::cout << "*******************PREDICCION*******************\n"
            << p << '\n';

        return p;
    }

    
};

#endif