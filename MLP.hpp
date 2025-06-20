#pragma once
#include <functional>
#include <fstream>
#include <sstream>
#include <tuple>
#include <chrono>
#include "MLPLayer.hpp"
#include "utils_mlp.hpp"

class MLP {
private:
    std::vector<MLPLayer> hidden_layers;
    std::mt19937 gen;
    MLPHyperparameters h;

    std::function<Matrix<double>(const Matrix<double>&)> factivation_soft;
    std::function<double(const double&)> factivation;
    std::function<double(const double&)> factivation_deri;

    void optimizate(const std::vector<Matrix<double>>& _nabla_w_acc,
        const std::vector<Matrix<double>>& _nabla_b_acc, const size_t& _batch) {

        const auto LAYERS = hidden_layers.size();
        
        if (h.optimizer == Optimizer::RMS_PROP) {
            const double LR = h.learning_rate / static_cast<double>(_batch);
            for (size_t i = 0; i < LAYERS; ++i) {
                auto& layer = hidden_layers[i];

                layer.cache_w = layer.cache_w * h.decay_rate + _nabla_w_acc[i].apply_function(square) * (1.0 - h.decay_rate);
                layer.cache_b = layer.cache_b * h.decay_rate + _nabla_b_acc[i].apply_function(square) * (1.0 - h.decay_rate);

                Matrix<double> adjusted_w = _nabla_w_acc[i];
                Matrix<double> adjusted_b = _nabla_b_acc[i];

                for (size_t r = 0; r < adjusted_w.rows(); ++r)
                    for (size_t c = 0; c < adjusted_w.cols(); ++c)
                        adjusted_w[r][c] /= std::sqrt(layer.cache_w[r][c] + h.epsilon);

                for (size_t r = 0; r < adjusted_b.rows(); ++r)
                    adjusted_b[r][0] /= std::sqrt(layer.cache_b[r][0] + h.epsilon);

                layer.weights = layer.weights - adjusted_w * LR;
                layer.biases = layer.biases - adjusted_b * LR;
            }
        }
        else if (h.optimizer == Optimizer::ADAM) {
            for (size_t i = 0; i < LAYERS; ++i) {
                auto& layer = hidden_layers[i];

                layer.m_w = layer.m_w * h.beta1 + _nabla_w_acc[i] * (1 - h.beta1);
                layer.v_w = layer.v_w * h.beta2 + _nabla_w_acc[i].apply_function(square) * (1.0 - h.beta2);

                layer.m_b = layer.m_b * h.beta1 + _nabla_b_acc[i] * (1 - h.beta1);
                layer.v_b = layer.v_b * h.beta2 + _nabla_b_acc[i].apply_function(square) * (1.0 - h.beta2);

                const double LR_T = h.learning_rate * std::sqrt(1.0 - std::pow(h.beta2, h.timestep)) / (1.0 - std::pow(h.beta1, h.timestep));

                for (size_t r = 0; r < layer.weights.rows(); ++r)
                    for (size_t c = 0; c < layer.weights.cols(); ++c)
                        layer.weights[r][c] -= LR_T * layer.m_w[r][c] / (std::sqrt(layer.v_w[r][c]) + h.epsilon);

                for (size_t r = 0; r < layer.biases.rows(); ++r)
                    layer.biases[r][0] -= LR_T * layer.m_b[r][0] / (std::sqrt(layer.v_b[r][0]) + h.epsilon);
            }
            ++h.timestep;
        }
        else {
            const double LR = h.learning_rate / static_cast<double>(h.batch);
            for (size_t i = 0; i < LAYERS; ++i) {
                auto& layer = hidden_layers[i];
                layer.weights = layer.weights - (_nabla_w_acc[i] * LR);
                layer.biases = layer.biases - (_nabla_b_acc[i] * LR);
            }
        }
    }

    Matrix<double> forward(const Matrix<double>& _input) {
        Matrix<double> a = _input;
        std::uniform_real_distribution<> dropout_dist(0.0, 1.0);

        auto begin = hidden_layers.begin();
        auto end = hidden_layers.end();

        for (; begin != end; ++begin) {
            auto& layer = *begin;

            layer.z = (layer.weights * a) + layer.biases;

            if (begin != end - 1) {
                layer.activation = layer.z.apply_function(factivation);
            }
            else {
                layer.activation = factivation_soft(layer.z);
            }
                
            a = layer.activation;
        }

        return a;
    }

    auto backprop(const Matrix<double>& x, const Matrix<double>& y) {

        std::vector<Matrix<double>> nabla_w(hidden_layers.size());
        std::vector<Matrix<double>> nabla_b(hidden_layers.size());

        Matrix<double> delta_i = hidden_layers.back().activation - y;
  
        nabla_w.back() = delta_i.outter_product(hidden_layers[hidden_layers.size() - 2].activation);
        nabla_b.back() = delta_i;
        const auto LAYERS = static_cast<int>(hidden_layers.size() - 2);

        for (int l = LAYERS; l >= 0; --l) {
            Matrix<double> sp = hidden_layers[l].z.apply_function(factivation_deri);
            Matrix<double> wTp = hidden_layers[l + 1].weights.transpose() * delta_i;

            for (size_t i = 0; i < wTp.rows(); ++i)
                wTp[i][0] *= sp[i][0];

            delta_i = wTp;
            Matrix<double> prev_activation = (l == 0) ? x : hidden_layers[l - 1].activation;
            nabla_b[l] = delta_i;
            nabla_w[l] = delta_i.outter_product(prev_activation);
        }

        return std::make_tuple(nabla_w, nabla_b);
    }

    double update_mini_batch(const std::vector<Matrix<double>>& _batch_x,
        const std::vector<Matrix<double>>& _batch_y) {

        const auto LAYERS = hidden_layers.size();
        const auto BATCH = _batch_x.size();
        double total_loss = 0;

        std::vector<Matrix<double>> nabla_w_acc(LAYERS);
        std::vector<Matrix<double>> nabla_b_acc(LAYERS);

        for (size_t i = 0; i < LAYERS; ++i) {
            nabla_w_acc[i] = Matrix<double>(hidden_layers[i].weights.rows(), hidden_layers[i].weights.cols(), 0.0);
            nabla_b_acc[i] = Matrix<double>(hidden_layers[i].biases.rows(), 1, 0.0);
        }

        for (size_t i = 0; i < BATCH; ++i) {
            total_loss += cross_entropy_loss(forward(_batch_x[i]), _batch_y[i]);

            const auto [delta_nabla_w, delta_nabla_b] = backprop(_batch_x[i], _batch_y[i]);

            for (size_t j = 0; j < LAYERS; ++j) {
                nabla_w_acc[j] = nabla_w_acc[j] + delta_nabla_w[j];
                nabla_b_acc[j] = nabla_b_acc[j] + delta_nabla_b[j];
            }
        }

        optimizate(nabla_w_acc, nabla_b_acc, BATCH);

        return total_loss;
    }


    void log_header(const std::string& _LOSS_FILE_NAME,
        const std::string& _ACCURACY_FILE_NAME,
        const std::string& _ELAPSED_FILE_NAME,
        const std::string& _LOG_FILE_NAME,
        const size_t& _TRAINING_SIZE) {

        const auto LAYERS_SIZE = hidden_layers.size();

        std::ofstream loss_file(_LOSS_FILE_NAME, std::ofstream::trunc);
        loss_file << "epoca,perdida\n";
        loss_file.close();

        std::ofstream accuracy_file(_ACCURACY_FILE_NAME, std::ofstream::trunc);
        accuracy_file << "epoca,precision\n";
        accuracy_file.close();

        std::ofstream elapsed_file(_ELAPSED_FILE_NAME, std::ofstream::trunc);
        accuracy_file << "epoca,minutos\n";
        accuracy_file.close();

        std::ofstream log_file(_LOG_FILE_NAME, std::ofstream::trunc);

        log_file << "DATOS DE ENTRENAMIENTO: " << _TRAINING_SIZE << '\n'
            << "BATCH: " << h.batch << '\n'
            << "TAZA APRENDIZAJE: " << h.learning_rate << '\n'
            << "ORDENAMIENTO: " << h.shuffle << '\n'
            << "EPOCAS: " << h.epochs << '\n';

        log_file << "PESOS Y BIASES INICIALES: \n";
        for (size_t l = 0; l < LAYERS_SIZE; ++l) {
            log_file << "CAPA " << l << ": \n"
                << "PESOS\n" << hidden_layers[l].weights << "\n\n"
                << "BIASES\n" << hidden_layers[l].biases << "\n\n";
        }
        log_file.close();
    }

    void log(const std::string& _LOSS_FILE_NAME,
        const std::string& _ACCURACY_FILE_NAME,
        const std::string& _ELAPSED_FILE_NAME,
        const std::string& _LOG_FILE_NAME,
        const size_t& _EPOCH,
        const double& _LOSS,
        const double& _ACCURACY,
        const float& _ELAPSED) {
        
        const auto LAYERS_SIZE = hidden_layers.size();
        
        std::ofstream loss_file(_LOSS_FILE_NAME, std::ofstream::app);
        std::ofstream elapsed_file(_ELAPSED_FILE_NAME, std::ofstream::app);
        std::ofstream accuracy_file(_ACCURACY_FILE_NAME, std::ofstream::app);
        std::ofstream log_file(_LOG_FILE_NAME, std::ofstream::app);

        loss_file << _EPOCH << ',' << _LOSS << '\n';
        loss_file.close();

        accuracy_file << _EPOCH << ',' << _ACCURACY << '\n';
        accuracy_file.close();

        elapsed_file << _EPOCH << ',' << _ELAPSED << '\n';
        elapsed_file.close();

        log_file << "EPOCA: " << _EPOCH << " PESOS Y BIASES : \n";
        for (size_t l = 0; l < LAYERS_SIZE; ++l) {
            log_file << "CAPA " << l << ": \n"
                << "PESOS\n" << hidden_layers[l].weights << "\n\n"
                << "BIASES\n" << hidden_layers[l].biases << "\n\n";
        }
        log_file.close();
    }


public:

    MLP() = default;

    explicit MLP(const MLPHyperparameters& _h) : gen(std::random_device{}()), h(_h) {

        switch (_h.activation) {
            case Activation::SIGMOID:
                factivation = sigmoid;
                factivation_deri = sigmoid_deri;
                break;
            case Activation::RELU:
                factivation = relu;
                factivation_deri = relu_deri;
                break;
            case Activation::TANH:
                factivation = tanhm;
                factivation_deri = tanh_deri;
                break;
        }
          
        factivation_soft = softmax;

        const auto LAYERS_SIZE = h.layers.size();
        const auto& layers = h.layers;

        for (size_t i = 1; i < LAYERS_SIZE; ++i)
            hidden_layers.emplace_back(layers[i - 1], layers[i], h.initializer, h.bias_init, gen);
    }

    explicit MLP(const std::string& _file_name, const MLPHyperparameters& _h)
        : gen(std::random_device{}()), h(_h) {
            
        std::ifstream in(_file_name + ".dat", std::ios::binary);

        size_t size{};
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        hidden_layers.reserve(size);
      
        for (size_t i = 0; i < size; ++i) {      
            MLPLayer layer;
            in >> layer.weights >> layer.biases;
            hidden_layers.emplace_back(layer);
        }

        in.close();
    }

    void train(const std::vector<Matrix<double>>& _training_data_x,
        const std::vector<Matrix<double>>& _training_data_y,
        const std::vector<Matrix<double>>& _testing_data_x,
        const std::vector<Matrix<double>>& _testing_data_y) {

        if (_training_data_x.size() != _training_data_y.size()
            || _testing_data_x.size() != _testing_data_y.size())
            throw std::invalid_argument("E");
                
        auto get_random_name = [](std::mt19937& _gen, const size_t& _len) {
            std::stringstream ss;
            std::uniform_int_distribution<> dist(48, 57);

            for (size_t i = 0; i < _len; ++i)
                ss << static_cast<char>(dist(_gen));

            return ss.str();
            };

        const auto TRAINING_SIZE = _training_data_x.size();

        const auto LOSS_FILE_NAME = get_random_name(gen, 6) + "-mlp-loss.csv";
        const auto ACCURACY_FILE_NAME = get_random_name(gen, 6) + "-mlp-accuracy.csv";
        const auto ELAPSED_FILE_NAME = get_random_name(gen, 6) + "-mlp-elapsed.csv";
        const auto LOG_FILE_NAME = get_random_name(gen, 6) + "-mlp-logger.log";
        double total_loss = 0;

        std::vector<size_t> random_indices(TRAINING_SIZE, 0);
        for (size_t j = 1; j < TRAINING_SIZE; ++j) random_indices[j] = j;

        if (h.debug) log_header(LOSS_FILE_NAME, ACCURACY_FILE_NAME, ELAPSED_FILE_NAME, LOG_FILE_NAME, TRAINING_SIZE);

        for (size_t e = 1; e <= h.epochs; ++e) {
            if (h.shuffle) std::shuffle(random_indices.begin(), random_indices.end(), gen);

            std::cout << "EPOCA: " << e << '\n';
  
            auto begin_t = std::chrono::high_resolution_clock::now();
            for (size_t begin = 0, end = 0; begin < TRAINING_SIZE; begin += h.batch) {
                end = std::min(begin + h.batch, TRAINING_SIZE);

                std::vector<Matrix<double>> batch_x;
                std::vector<Matrix<double>> batch_y;

                for (size_t j = begin; j < end; ++j) {
                    batch_x.push_back(_training_data_x[random_indices[j]]);
                    batch_y.push_back(_training_data_y[random_indices[j]]);
                }

                total_loss += update_mini_batch(batch_x, batch_y);
            }
            auto end_t = std::chrono::high_resolution_clock::now();

            auto elapsed_t = std::chrono::duration_cast<std::chrono::seconds>(end_t - begin_t).count();
            auto accuracy = get_accuracy(_testing_data_x, _testing_data_y);

            if (h.debug) log(LOSS_FILE_NAME, ACCURACY_FILE_NAME, ELAPSED_FILE_NAME, LOG_FILE_NAME, e, total_loss / static_cast<double>(TRAINING_SIZE), accuracy, static_cast<float>(elapsed_t) / 60.f);

            total_loss = 0;
        }
    }

    double get_accuracy(const std::vector<Matrix<double>>& _testing_data_x,
        const std::vector<Matrix<double>>& _testing_data_y) {

        int correct = 0;
        const auto TESTING_SIZE = _testing_data_x.size();

        for (size_t i = 0; i < TESTING_SIZE; ++i) {

            auto y_pre = predict(_testing_data_x[i]);

            size_t predicted = 0;
            double max_prob = y_pre[0][0];
            for (size_t j = 1; j < y_pre.rows(); ++j) {
                if (y_pre[j][0] > max_prob) {
                    max_prob = y_pre[j][0];
                    predicted = j;
                }
            }

            size_t actual = 0;
            for (size_t j = 0; j < _testing_data_y[i].rows(); ++j) {
                if (_testing_data_y[i][j][0] == 1.0) {
                    actual = j;
                    break;
                }
            }

            if (predicted == actual)
                ++correct;
        }

        return static_cast<double>(correct) / static_cast<double>(TESTING_SIZE);
    }

    Matrix<double> predict(const Matrix<double>& x) {
        return forward(x);
    }

    void save_weights(const std::string& _file_name) const {
        std::ofstream out(_file_name + ".dat", std::ofstream::trunc | std::ios::binary);
        size_t size = hidden_layers.size();

        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& l : hidden_layers) 
            out << l.weights << l.biases;                        
        out.close();
    }

};