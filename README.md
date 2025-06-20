# Convolutional Neural Network (CNN) en C++ Moderno

Este proyecto implementa una red neuronal convolucional (CNN) sencilla y modular en **C++ moderno**, con énfasis en claridad y encapsulamiento.

## Características

- Operaciones convolucionales multicanal (soporte para RGB)
- Parámetros personalizables: `stride`, `padding`, `kernel`
- Activación ReLU
- Pooling (`max`, `avg`, `min`)
- Aplanamiento (`flatten`) para conexión con redes densas (MLP)
- Cálculo automático del tamaño de entrada al MLP

## Estructura

### `Matrix<T>`

Clase genérica de matriz con soporte para:

- Acceso tipo `m[i][j]`
- Operaciones aritméticas
- Transpuesta y aplanamiento
- Serialización binaria

### `ConvLayer<T>`

Encapsula una capa convolucional con:

- Múltiples filtros (cada uno con kernels por canal)
- Activación ReLU
- Pooling (opcional)
- Parámetros por capa: `stride`, `padding`, `pool_size`, `pool_mode`

### CNN<T>

Encapsula una red con múltiples capas convolucionales.

### `MLP`
- Inicializacion personalizada `Xavier`, `He`, `Random`
- Activacion personalizada `sigmoid`, `relu`, `tanh`
- Softmax

### calculateMLPInputSize(...)
Calcula dinámicamente el número de entradas que debe recibir la capa MLP tras aplicar todas las capas convolucionales:
```cpp
size_t entradas = calculateMLPInputSize(altura, ancho, canales, capas);
```

### Requisitos
- C++17 o superior
- Compilador compatible con STL (g++, clang++, etc.)

### Compilación

```bash
g++ -std=c++17 main.cpp -o cnn
./cnn
```
