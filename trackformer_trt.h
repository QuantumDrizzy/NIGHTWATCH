#pragma once

#include <vector>
#include <string>
#include <iostream>

// Si tienes instalado el SDK de TensorRT de Nvidia, 
// descomenta esto en el futuro cuando compilemos con nvinfer.lib
// #define NIGHTWATCH_USE_TENSORRT
// #include <NvInfer.h>

extern "C" {

// Tracklet seed: one detection produced per frame by the inference step
// (the TensorRT network when compiled with NIGHTWATCH_USE_TENSORRT, otherwise
// the classical CFAR-centroid fallback). Consumed by the Kalman gate in main.cpp.
struct TrackletSeed {
    float x;            // Coordenada X sub-pixel detectada
    float y;            // Coordenada Y sub-pixel detectada
    float vx;           // Componente de velocidad en X
    float vy;           // Componente de velocidad en Y
    float confidence;   // Probabilidad de detección (p_det ∈ [0,1])
    unsigned int timestamp; // Índice de frame temporal
    float entropy;      // Entropía de Shannon del mapa de atención
};

}

class TrackformerTRT {
public:
    TrackformerTRT();
    ~TrackformerTRT();

    // Inicializa el motor de inferencia con el archivo .engine
    bool initialize(const std::string& engine_path);

    // Ingiere el Cubo Diferencial de 16 frames y devuelve una semilla
    // En producción, cube_data apunta a la memoria en la GPU (d_density)
    TrackletSeed infer(float* d_cube_data_16frames, int width, int height, unsigned int current_frame);

private:
#ifdef NIGHTWATCH_USE_TENSORRT
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
#endif
    bool is_initialized;
};
