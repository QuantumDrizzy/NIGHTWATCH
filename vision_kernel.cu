#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "nightwatch_vision.h"
__global__ void process_vision_kernel(unsigned short* raw_ir, float* density_matrix, float* prev_density_matrix, int width, int height, float gain, bool ifu_mode, float denoise_threshold, VelocityData* v_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float val = (float)raw_ir[idx] / 1024.0f;

        if (val < denoise_threshold) val = 0.0f;

        float current_prob = (density_matrix[idx] * 0.90f) + (val * gain);
        density_matrix[idx] = current_prob;

        // --- CALCULO DE NITIDEZ (Laplaciano para Lucky Imaging) ---
        // Solo hilos internos para el stencil
        float laplacian = 0;
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            laplacian = 4.0f * val 
                      - (float)raw_ir[idx + 1] / 1024.0f 
                      - (float)raw_ir[idx - 1] / 1024.0f 
                      - (float)raw_ir[idx + width] / 1024.0f 
                      - (float)raw_ir[idx - width] / 1024.0f;
        }

        // NOTE: v_out is zeroed by the host (cudaMemset) before each launch.
        // An in-kernel reset by thread (0,0) + __syncthreads() would NOT be
        // safe — __syncthreads() only synchronizes within a block, so it would
        // race with atomicAdds from other blocks.
        float diff = current_prob - prev_density_matrix[idx];
        if (diff > 0.5f) {
            atomicAdd(&(v_out->dx), (float)(x - width/2) * diff * 0.0001f);
            atomicAdd(&(v_out->dy), (float)(y - height/2) * diff * 0.0001f);
            atomicAdd(&(v_out->energy), diff);
        }
        
        // Acumular varianza del laplaciano como score de nitidez
        atomicAdd(&(v_out->sharpness), laplacian * laplacian);

        prev_density_matrix[idx] = current_prob;
    }
}

// FASE 6.2: CELL-AVERAGING CONSTANT FALSE ALARM RATE (CA-CFAR 2D)
__global__ void cfar_kernel(const float* in_matrix, float* out_matrix, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float cut = in_matrix[idx]; // Cell Under Test
        
        // Ventanas: 9x9 Entrenamiento (Radio 4), 3x3 Guarda (Radio 1)
        const int WINDOW_SIZE = 4;
        const int GUARD_SIZE = 1;
        
        // Bordes a 0
        if (x < WINDOW_SIZE || x >= width - WINDOW_SIZE || y < WINDOW_SIZE || y >= height - WINDOW_SIZE) {
            out_matrix[idx] = 0.0f;
            return;
        }

        float noise_sum = 0.0f;
        int num_cells = 0;

        for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; j++) {
            for (int i = -WINDOW_SIZE; i <= WINDOW_SIZE; i++) {
                if (abs(i) <= GUARD_SIZE && abs(j) <= GUARD_SIZE) continue; // Ignorar guarda y CUT
                int sample_idx = (y + j) * width + (x + i);
                noise_sum += in_matrix[sample_idx];
                num_cells++;
            }
        }

        float noise_level = noise_sum / (float)num_cells;
        
        // Alpha (Umbral de Detección CFAR)
        // En un entorno militar se calcula usando la probabilidad Pfa, aquí lo fijamos empíricamente.
        const float ALPHA = 3.5f; 
        float threshold = noise_level * ALPHA;
        
        // Binarizar/Filtrar
        if (cut > threshold && cut > 0.05f) {
            out_matrix[idx] = cut; // Sobrevive al CFAR
        } else {
            out_matrix[idx] = 0.0f; // Aniquilado
        }
    }
}

extern "C" void launch_vision_kernel(unsigned short* raw_ir, float* density_matrix, float* prev_density_matrix, int width, int height, float gain, bool ifu_mode, float denoise_threshold, VelocityData* v_out) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    process_vision_kernel<<<grid, block>>>(raw_ir, density_matrix, prev_density_matrix, width, height, gain, ifu_mode, denoise_threshold, v_out);
}

extern "C" void launch_cfar_kernel(const float* in_matrix, float* out_matrix, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    cfar_kernel<<<grid, block>>>(in_matrix, out_matrix, width, height);
}
