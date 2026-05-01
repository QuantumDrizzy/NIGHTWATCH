#include "trackformer_trt.h"
#include <cmath>
#include <iostream>

TrackformerTRT::TrackformerTRT() : is_initialized(false) {
#ifdef NIGHTWATCH_USE_TENSORRT
    runtime = nullptr;
    engine = nullptr;
    context = nullptr;
#endif
}

TrackformerTRT::~TrackformerTRT() {
#ifdef NIGHTWATCH_USE_TENSORRT
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
#endif
}

bool TrackformerTRT::initialize(const std::string& engine_path) {
    // Aquí cargaremos el archivo INT8 ONNX transformado a .engine
    // Por ahora, simulamos la inicialización exitosa.
    std::cout << "[TRACKFORMER] Inicializando motor TensorRT desde: " << engine_path << std::endl;
    is_initialized = true;
    return true;
}

TrackletSeed TrackformerTRT::infer(float* d_cube_data_16frames, int width, int height, unsigned int current_frame) {
    TrackletSeed seed = {0};
    seed.timestamp = current_frame;

    if (!is_initialized) {
        seed.confidence = 0.0f;
        return seed;
    }

#ifdef NIGHTWATCH_USE_TENSORRT
    // Aquí irían las llamadas:
    // context->enqueueV2(buffers, stream, nullptr);
    // cudaMemcpyAsync(output_host, output_device, ...);
#else
    // SIMULACIÓN DE INFERENCIA DE LA RED NEURONAL (DUMMY)
    // Para probar el flujo C++ sin romper la compilación si no hay TensorRT.
    // Simulamos que la red detecta algo en el centro con ruido.
    
    // (En la vida real, estos valores vendrían de la salida FP16/INT8 de la red)
    seed.x = (width / 2.0f) + (std::rand() % 10 - 5);
    seed.y = (height / 2.0f) + (std::rand() % 10 - 5);
    seed.vx = 1.2f;
    seed.vy = 0.5f;
    seed.confidence = 0.85f; // Alta confianza simulada
    seed.entropy = 0.4f;     // Baja entropía (foco claro)
#endif

    return seed;
}
