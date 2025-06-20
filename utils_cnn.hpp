#include <string>
#include <tuple>

enum class PoolMode { NONE, MIN, MAX, AVG };

struct ConvLayeryperparameters {
	bool relu = false;
	bool pooling = false;
	size_t padding = 0;
	size_t stride = 0;
	size_t pool_size = 0;
	PoolMode pool_mode = PoolMode::NONE;
    bool debug = false;
};

auto get_test_image() {
    std::vector<double> canalR;
    std::vector<double> canalG;
    std::vector<double> canalB;
    int filas{};
    int columnas{};
   
    std::ifstream archivo("pic_0010.bin", std::ios::binary);
    if (!archivo) {
        std::cerr << "Error al abrir el archivo para lectura." << std::endl;
        throw std::invalid_argument("e");
    }

    // Leer dimensiones
    archivo.read(reinterpret_cast<char*>(&filas), sizeof(int));
    archivo.read(reinterpret_cast<char*>(&columnas), sizeof(int));
    int totalPixeles = filas * columnas;

    canalR.resize(totalPixeles);
    canalG.resize(totalPixeles);
    canalB.resize(totalPixeles);
    uint8_t aux;

    for (int i = 0; i < totalPixeles; ++i) {

        archivo.read(reinterpret_cast<char*>(&aux), 1);

        canalR[i] = static_cast<double>(aux) / 255.0;

        archivo.read(reinterpret_cast<char*>(&aux), 1);

        canalG[i] = static_cast<double>(aux) / 255.0;

        archivo.read(reinterpret_cast<char*>(&aux), 1);

        canalB[i] = static_cast<double>(aux) / 255.0;
    }


    return std::make_tuple(canalR, canalG, canalB);
}