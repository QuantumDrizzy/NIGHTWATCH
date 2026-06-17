/*
 * nightwatch_vision.h — Shared types for nightwatch-vision pipeline
 * ─────────────────────────────────────────────────────────
 * Plain C structs so this header can be included from both
 * CUDA translation units and host C++ code without issues.
 */

#pragma once

// ── Sensor resolution ────────────────────────────────────────────────────────
#define NIGHTWATCH_WIDTH    512
#define NIGHTWATCH_HEIGHT   424

// ── Blob detection parameters ────────────────────────────────────────────────
#define MAX_BLOBS          8       // max simultaneous tracked blobs
#define BLOB_GRID_COLS     8       // spatial grid columns (8 × 8 = 64 cells)
#define BLOB_GRID_ROWS     8
#define BLOB_GRID_N        64

// Default thresholds — override in main() for live-sensor vs synthetic mode
#define BLOB_THRESHOLD_DEFAULT  1.0f    // density matrix value
#define BLOB_MIN_WEIGHT_DEFAULT 20.0f   // Σ(w·pixel) per cell

// ── Synthetic scene ──────────────────────────────────────────────────────────
#define MAX_SCENE_OBJECTS  4

/*
 * BlobResult — output of launch_blob_detection() per detected blob.
 * Coordinates are in pixel space of the density matrix.
 */
typedef struct {
    float cx;       /* centroid x [px]                          */
    float cy;       /* centroid y [px]                          */
    float weight;   /* Σ(intensity above threshold) — confidence */
    float radius;   /* √(pixel_count / π) — size estimate [px]  */
    int   valid;    /* 1 = active blob, 0 = empty slot           */
} BlobResult;

/*
 * SceneObject — one IR-reflective object in the synthetic scene.
 */
typedef struct {
    float cx, cy;       /* centre in image pixels                */
    float vx, vy;       /* velocity [px / frame]  (host-updated) */
    float distance;     /* sensor distance [m] ∈ [0.3, 4.0]     */
    float reflectance;  /* surface reflectance ∈ [0, 1]          */
    float radius_m;     /* physical radius [m]                   */
    int   active;       /* 1 = render this object                */
} SceneObject;

struct VelocityData {
    float dx;
    float dy;
    float energy;
    float sharpness;
};

// ── CUDA kernel APIs (extern "C" linkage) ────────────────────────────────────
#ifdef __cplusplus
extern "C" {
#endif

/* Existing pipeline kernel — vision_kernel.cu */
void launch_vision_kernel(unsigned short* raw_ir,
                          float*          density_matrix,
                          float*          prev_density_matrix,
                          int width, int height,
                          float gain, bool ifu_mode,
                          float denoise_threshold,
                          struct VelocityData* v_out);

/* CA-CFAR 2D Kernel */
void launch_cfar_kernel(const float* in_matrix, float* out_matrix, int width, int height);

/*
 * Grid-based blob detection — vision_kernel.cu
 *
 * d_results      : device ptr, at least MAX_BLOBS × sizeof(BlobResult)
 * d_result_count : device ptr to int; receives number of blobs found
 *
 * Caller must zero d_results / d_result_count before each call
 * (handled internally by launch_blob_detection).
 */
void launch_blob_detection(const float* density_matrix,
                           int   width, int height,
                           float threshold,
                           float min_weight,
                           BlobResult* d_results,
                           int*        d_result_count,
                           int         max_blobs);

/*
 * Realistic time-of-flight (ToF) IR frame generator — synth_tof.cu
 *
 * Noise model:
 *   thermal floor : μ=327 ADU, σ=25 ADU  (ToF sensor empirical)
 *   multipath     : sinusoidal ripple ±18 ADU
 *   shot noise    : σ = √(signal/16)   (photon counting ToF)
 *   saturation    : specular clipping above ~52 000 ADU
 *   falloff       : I ∝ reflectance / (1 + distance²)
 *
 * h_objects[n_objects] is a host-side array; the function copies it to
 * device constant memory before launching.
 */
void launch_synth_frame(unsigned short*    d_out,
                        int                width,
                        int                height,
                        const SceneObject* h_objects,
                        int                n_objects,
                        unsigned int       frame_id);

#ifdef __cplusplus
}
#endif
