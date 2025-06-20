#include "CNN.hpp"

template <typename T>
size_t calculateMLPInputSize(
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    const std::vector<ConvLayer<T>>& conv_layers
) {
    size_t height = input_height;
    size_t width = input_width;
    size_t channels = input_channels;

    for (const auto& layer : conv_layers) {
        // Asumimos que todos los kernels son cuadrados y tienen mismo tamaño
        const auto& filters = layer.getFilters();
        if (filters.empty() || filters[0].empty())
            throw std::runtime_error("Capa convolucional sin filtros validos");

        size_t kernel_size = filters[0][0].rows();
        size_t stride = layer.getStride();
        size_t padding = layer.getPadding();
        bool pooling = layer.usesPooling();
        size_t pool_size = layer.getPoolSize();

        height = (height + 2 * padding - kernel_size) / stride + 1;
        width = (width + 2 * padding - kernel_size) / stride + 1;

        if (pooling) {
            height = height / pool_size;
            width = width / pool_size;
        }

        // Numero de mapas de salida = numero de filtros
        channels = filters.size();
    }

    return height * width * channels;
}


int main() {

    const size_t size = 28;

    // Imagen RGB
    auto [a, b, c] = get_test_image();
        
    auto R = Matrix(a, a.size(), 28, 28);
    auto G = Matrix(b, b.size(), 28, 28);
    auto B = Matrix(c, c.size(), 28, 28);

    std::vector<Matrix<double>> rgb = { R, G, B };

    // Kernels
    std::vector<Matrix<double>> kernel1 = {
        Matrix<double>({{1,0,-1},{1,0,-1},{1,0,-1}}),
        Matrix<double>({{0,1,0},{0,1,0},{0,1,0}}),
        Matrix<double>({{-1,-1,1},{0,0,0},{1,1,-1}})
    };

    std::vector<Matrix<double>> kernel2 = {
        Matrix<double>({{-1,-2,-1},{0,0,0},{1,2,1}}),
        Matrix<double>({{1,0,-1},{2,0,-2},{1,0,-1}}),
        Matrix<double>({{0,1,0},{1,-4,1},{0,1,0}})
    };

    // Hiperparametros de la capa convolucional
    ConvLayeryperparameters h_cvl;
    h_cvl.padding = 1;
    h_cvl.stride = 1;
    h_cvl.pool_size = 2;
    h_cvl.pool_mode = PoolMode::MAX;
    h_cvl.relu = true;
    h_cvl.pooling = true;
    h_cvl.debug = false;

    ConvLayer<double> layer1({ kernel1, kernel2 }, h_cvl);

    std::vector<ConvLayer<double>> layers = { layer1 };

    // Calcular entrada del MLP
    auto i = calculateMLPInputSize<double>(size, size, 3, layers);

    MLPHyperparameters h_mlp;
    h_mlp.layers = { i, 32, 16, 10 };
    h_mlp.activation = Activation::RELU;
    h_mlp.initializer = Initializer::HE;

    auto mlp = MLP(h_mlp);

    CNN<double> cnn(layers, mlp);

    auto outputs = cnn.forward(rgb);
    
    return 0;
}

