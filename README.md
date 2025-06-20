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

### Capturas
- Teniendo como entrada una imagen rgb de 8x8
- 2 kernels de 3x3 para rgb
- padding = 1
- stride = 1
- pool size = 2
- pool mode = max
- relu = true
- MLP con 10 neuronas de salida

![1](https://github.com/user-attachments/assets/2d6c8f6a-fd92-432f-a535-257494192edc)

![2](https://github.com/user-attachments/assets/b470d14f-0fee-4e69-8281-5537bfd4b371)

![3](https://github.com/user-attachments/assets/4aac4793-6c58-4780-a7fa-d19a515a4281)

![4](https://github.com/user-attachments/assets/8ca1742f-9324-4651-911a-ee066ff892c2)

![5](https://github.com/user-attachments/assets/df3138e2-26df-433e-9bdb-37978ff757b8)


