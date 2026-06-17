/*
 * synth_tof.cu — Realistic time-of-flight (ToF) IR frame generator
 * ─────────────────────────────────────────────────────────────
 * Simulates the noise profile of a generic ToF depth/IR sensor without
 * physical hardware.  Useful for testing the CUDA vision pipeline and
 * blob detection offline, independent of which sensor is wired in for
 * live capture.
 *
 * Noise model (parameters from ToF sensor characterisation
 * literature — Sarbolandi et al. 2015, Dal Mutto et al. 2012):
 *
 *   Thermal floor   : μ = 327 ADU, σ = 25 ADU  (read noise + dark current)
 *   Multipath        : sinusoidal ripple ±18 ADU (multi-bounce reflection)
 *   Shot noise       : σ_shot = √(signal/16)    (photon Poisson process)
 *   Distance falloff : I ∝ reflectance / (1 + d²)  (inverse-square, d in m)
 *   Saturation       : specular pixels clip stochastically above ~52 000 ADU
 *   Quantisation     : round-to-nearest UINT16
 *
 * RNG — LCG (Numerical Recipes constants) per pixel + frame seed.
 * Box-Muller for Gaussian noise — no cuRAND dependency.
 *
 * Scene objects are passed from the host as a small struct array;
 * the kernel reads them from CUDA constant memory (fast broadcast).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "nightwatch_vision.h"

// ── ToF IR optics ─────────────────────────────────────────────────────────────
// Focal length of the IR projector in pixels (representative ToF calibration)
#define TOF_FOCAL_PX   364.0f

// Noise constants — Sarbolandi et al. (2015)
#define THERMAL_FLOOR_ADU  327.0f
#define THERMAL_SIGMA_ADU   25.0f
#define MULTIPATH_AMP       18.0f
#define SAT_ONSET_ADU    52000.0f
#define ADU_MAX          65535.0f

// ── Constant memory for scene objects (broadcast, no L1 pressure) ─────────────
static __constant__ SceneObject d_scene[MAX_SCENE_OBJECTS];

// ═══════════════════════════════════════════════════════════════════════════════
// Device-side LCG random number generator
// ═══════════════════════════════════════════════════════════════════════════════
// Knuth / Numerical Recipes LCG.  Each pixel carries its own state seeded
// from (pixel_index × prime + frame_seed × prime2) to ensure decorrelation.

__device__ __forceinline__ unsigned int lcg_next(unsigned int& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

// Uniform float in (0, 1) — upper 24 bits for best quality
__device__ __forceinline__ float lcg_u01(unsigned int& s) {
    return (float)(lcg_next(s) >> 8) * 5.96046448e-8f;   // / 2^24
}

// Zero-mean unit-Gaussian via Box-Muller (uses two LCG draws)
__device__ float lcg_gaussian(unsigned int& s) {
    float u = lcg_u01(s) + 1e-7f;   // guard against log(0)
    float v = lcg_u01(s);
    return sqrtf(-2.0f * logf(u)) * cosf(6.28318530f * v);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main frame generation kernel
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void generate_tof_frame_kernel(
    unsigned short* __restrict__ out,
    int width, int height,
    int n_objects,
    unsigned int frame_seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Per-pixel unique RNG state — two different primes to decorrelate x and y
    unsigned int rng = ((unsigned int)(y * width + x)) * 1000003u
                     ^ (frame_seed * 6291469u);
    // Warm up: discard first two values to break seed correlations
    lcg_next(rng);
    lcg_next(rng);

    // ── 1. Thermal noise floor ────────────────────────────────────────────────
    // Gaussian distributed around 327 ADU regardless of scene content.
    // This is the dominant noise source for ToF sensors at long distances.
    float val = THERMAL_FLOOR_ADU + THERMAL_SIGMA_ADU * lcg_gaussian(rng);

    // ── 2. Multipath interference ─────────────────────────────────────────────
    // Spatial low-frequency ripple from multi-bounce reflections off walls.
    // Frequency drifts slowly with frame index for realism.
    float phase_drift = (float)(frame_seed & 0xFF) * 0.0245f;   // ~0–6.3 rad
    val += MULTIPATH_AMP
         * sinf((float)x * 0.065f + phase_drift)
         * cosf((float)y * 0.055f);

    // ── 3. Scene object contributions ────────────────────────────────────────
    for (int k = 0; k < n_objects; ++k) {
        const SceneObject& obj = d_scene[k];
        if (!obj.active) continue;

        // Pixel-space sigma: object subtends (radius_m / distance) radians,
        // mapped to pixels through the IR focal length.
        float d_clamped = fmaxf(obj.distance, 0.3f);
        float sigma_px  = (obj.radius_m / d_clamped) * TOF_FOCAL_PX;
        sigma_px        = fmaxf(sigma_px, 1.5f);          // minimum 1.5 px

        float dx = (float)x - obj.cx;
        float dy = (float)y - obj.cy;
        float r2 = dx * dx + dy * dy;
        float s2 = sigma_px * sigma_px;

        // Skip pixels beyond 3σ — negligible contribution, saves sqrt
        if (r2 > 9.0f * s2) continue;

        // Distance-intensity falloff: I₀ / (1 + d²)  with d₀ = 1.0 m reference
        // Gives realistic 1/r² roll-off from the IR projector.
        float I0     = ADU_MAX * obj.reflectance;
        float peak   = I0 / (1.0f + d_clamped * d_clamped);

        // Gaussian spatial profile (far objects appear dimmer AND smaller)
        float signal = peak * expf(-0.5f * r2 / s2);

        // Shot noise: σ = √(signal/16)
        // The /16 factor models the photon-count averaging over 16 ToF pulses.
        float shot_sigma = sqrtf(signal * 0.0625f + 1.0f);
        signal += shot_sigma * lcg_gaussian(rng);

        // Specular saturation: bright surfaces partially clip to ADU_MAX.
        // Probability ramps linearly from 0 at SAT_ONSET to 30% at max.
        if (signal > SAT_ONSET_ADU) {
            float sat_prob = (signal - SAT_ONSET_ADU)
                           / (ADU_MAX - SAT_ONSET_ADU) * 0.30f;
            if (lcg_u01(rng) < sat_prob) {
                signal = ADU_MAX;
            }
        }

        val += signal;
    }

    // ── 4. Clamp + quantise to UINT16 ────────────────────────────────────────
    val = fmaxf(0.0f, fminf(val, ADU_MAX));
    out[y * width + x] = (unsigned short)(val + 0.5f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" void launch_synth_frame(
    unsigned short*    d_out,
    int                width,
    int                height,
    const SceneObject* h_objects,
    int                n_objects,
    unsigned int       frame_id)
{
    // Copy scene description to constant memory (tiny: 4 × ~40 B = 160 B)
    cudaMemcpyToSymbol(d_scene, h_objects,
                       (size_t)n_objects * sizeof(SceneObject));

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    generate_tof_frame_kernel<<<grid, block>>>(
        d_out, width, height, n_objects, frame_id);
}
