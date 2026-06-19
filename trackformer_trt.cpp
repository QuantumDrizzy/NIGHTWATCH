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
#ifdef NIGHTWATCH_USE_TENSORRT
    // TODO: deserialize engine_path into `engine`, create an execution context,
    // and allocate the I/O bindings. Until that lands, fail loudly rather than
    // pretend to be initialized.
    std::cerr << "[TRACKFORMER] NIGHTWATCH_USE_TENSORRT is defined but the engine "
                 "loader is not implemented yet (" << engine_path << ")." << std::endl;
    is_initialized = false;
    return false;
#else
    // No TensorRT compiled in: the pipeline runs the classical CFAR-centroid
    // fallback in infer(). This is a REAL detector (weighted centroid of the
    // CFAR mask), not a placeholder — it just doesn't classify like the net.
    std::cout << "[TRACKFORMER] TensorRT not compiled in — using classical "
                 "CFAR-centroid fallback (no neural classification)." << std::endl;
    is_initialized = true;
    return true;
#endif
}

TrackletSeed TrackformerTRT::infer(float* d_cfar_mask, int width, int height,
                                   unsigned int current_frame) {
    TrackletSeed seed = {0};
    seed.timestamp = current_frame;

    if (!is_initialized) {
        seed.confidence = 0.0f;
        return seed;
    }

#ifdef NIGHTWATCH_USE_TENSORRT
    // Real inference path (engine loaded in initialize()):
    //   context->enqueueV2(buffers, stream, nullptr);
    //   cudaMemcpyAsync(host_out, dev_out, ...);
    // For now this branch is unreachable because initialize() returns false
    // until the engine loader is implemented.
    seed.confidence = 0.0f;
    return seed;
#else
    // ── Classical fallback: weighted centroid of the CFAR-survived mask ──
    // d_cfar_mask is CUDA managed memory (cudaMallocManaged) and main.cpp
    // synchronizes before calling infer(), so the host can read it directly.
    // The CFAR kernel already zeroed everything that didn't survive, so the
    // centroid is the brightness-weighted centre of the detected blob.
    double sum_w = 0.0, sum_x = 0.0, sum_y = 0.0;
    int    n_active = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float w = d_cfar_mask[y * width + x];
            if (w > 0.0f) {
                sum_w += w;
                sum_x += (double)w * x;
                sum_y += (double)w * y;
                ++n_active;
            }
        }
    }

    if (sum_w > 0.0) {
        seed.x = (float)(sum_x / sum_w);
        seed.y = (float)(sum_y / sum_w);
        // Honest confidence heuristic: monotone in blob size, saturating at
        // ~25 surviving pixels. A real detection grows confidence; an empty
        // frame yields 0.0 (we do NOT fabricate a constant confidence).
        seed.confidence = fminf(1.0f, (float)n_active / 25.0f);
    } else {
        seed.x = width / 2.0f;
        seed.y = height / 2.0f;
        seed.confidence = 0.0f;   // nothing survived CFAR → no detection
    }

    // The classical fallback does not estimate velocity or attention entropy;
    // the Kalman filter derives velocity downstream. The neural path fills these.
    seed.vx = 0.0f;
    seed.vy = 0.0f;
    seed.entropy = 0.0f;
    return seed;
#endif
}
