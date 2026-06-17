#include <opencv2/opencv.hpp>
#include "nightwatch_vision.h"
#include "trackformer_trt.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <direct.h>

// VelocityData struct removed. Now in nightwatch_vision.h
extern "C" void launch_vision_kernel(unsigned short* raw_ir, float* density_matrix, float* prev_density_matrix, int width, int height, float gain, bool ifu_mode, float denoise_threshold, VelocityData* v_out);

struct KalmanState {
    float x, y, vx, vy, p;
};

void kalman_update(KalmanState& k, float mx, float my, float confidence = 1.0f) {
    // Prediction (Simplified)
    k.x = k.x + k.vx;
    k.y = k.y + k.vy;
    k.p = k.p + 0.1f; // Process noise
    
    // Update modulated by Network Confidence
    float k_gain = (k.p / (k.p + 2.0f)) * confidence; 
    
    k.x = k.x + k_gain * (mx - k.x);
    k.y = k.y + k_gain * (my - k.y);
    k.p *= (1.0f - k_gain);
}

int main(int argc, char** argv) {
    const int width = 640;
    const int height = 480;
    _mkdir("RAW_DATA");

    bool dataset_mode = false;
    bool live_simulate = false;
    int num_clips = 1;
    if (argc >= 2) {
        std::string mode(argv[1]);
        if (mode == "--generate-dataset" && argc >= 3) {
            dataset_mode = true;
            num_clips = std::atoi(argv[2]);
        } else if (mode == "--simulate-live") {
            live_simulate = true;
        }
    }

    unsigned short* d_raw_ir;
    float *d_density, *d_prev, *d_cfar;
    VelocityData* d_v_out;

    cudaMallocManaged(&d_raw_ir, width * height * sizeof(unsigned short));
    cudaMallocManaged(&d_density, width * height * sizeof(float));
    cudaMallocManaged(&d_prev, width * height * sizeof(float));
    cudaMallocManaged(&d_cfar, width * height * sizeof(float));
    cudaMallocManaged(&d_v_out, sizeof(VelocityData));

    memset(d_density, 0, width * height * sizeof(float));
    memset(d_prev, 0, width * height * sizeof(float));
    memset(d_cfar, 0, width * height * sizeof(float));

    KalmanState sat_kalman = {width/2.0f, height/2.0f, 0.0f, 0.0f, 1.0f};
    float orbital_phase = 0.0f;

    TrackformerTRT trackformer;
    trackformer.initialize("NIGHTWATCH_MOBILEVIT_XT.engine");

    std::cout << "[VANGUARDIA 4.1] I/O BINARIO + TOF NOISE MODEL ACTIVE" << std::endl;

    int total_frames = dataset_mode ? (num_clips * 16) : (live_simulate ? 999999999 : 200);
    
    std::ofstream telemetry_file;
    if (live_simulate) {
        telemetry_file.open("telemetry.jsonl", std::ios::app);
        std::cout << "[IPC] Escribiendo telemetria en telemetry.jsonl..." << std::endl;
    }

    for (int frame = 0; frame < total_frames; frame++) {
        // HYPERREALISTIC ToF SIMULATOR (synth_tof.cu)
        orbital_phase += 0.04f;
        float target_x = width/2 + 180.0f * cos(orbital_phase);
        float target_y = height/2 + 120.0f * sin(orbital_phase * 0.7f);
        
        // Ground truth velocity calculation (px/frame)
        float next_phase = orbital_phase + 0.04f;
        float vx = (width/2 + 180.0f * cos(next_phase)) - target_x;
        float vy = (height/2 + 120.0f * sin(next_phase * 0.7f)) - target_y;

        SceneObject objects[MAX_SCENE_OBJECTS] = {0};
        objects[0].cx = target_x;
        objects[0].cy = target_y;
        objects[0].vx = vx;
        objects[0].vy = vy;
        objects[0].distance = 1.5f;       // Satelite virtual a 1.5 metros
        objects[0].reflectance = 0.8f;    // Alta reflectancia NIR
        objects[0].radius_m = 0.02f;      // 2 centimetros de radio
        objects[0].active = 1;

        // Lanzar el simulador hiperrealista de ruido ToF de Claude
        launch_synth_frame(d_raw_ir, width, height, objects, 1, frame);

        // Procesar vision cuantica y tracking (Acumulacion Temporal)
        launch_vision_kernel(d_raw_ir, d_density, d_prev, width, height, 2.5f, false, 0.08f, d_v_out);
        cudaDeviceSynchronize();

        // Filtrado Espacial CFAR 2D (Aisla fuentes de calor)
        launch_cfar_kernel(d_density, d_cfar, width, height);
        cudaDeviceSynchronize();

        static float best_sharpness = 0;
        static int best_frame = 0;

        if (d_v_out->sharpness > best_sharpness) {
            best_sharpness = d_v_out->sharpness;
            best_frame = frame;
        }

        // --- FASE 4.3: INFERENCIA TENSORRT Y PUERTA DE MAHALANOBIS ---
        // Se alimenta con la mascara binarizada CFAR en lugar de la densidad bruta
        TrackletSeed seed = trackformer.infer(d_cfar, width, height, frame);
        
        // Calcular predicción actual del Kalman
        float pred_x = sat_kalman.x + sat_kalman.vx;
        float pred_y = sat_kalman.y + sat_kalman.vy;
        
        // Calcular Distancia de Mahalanobis simplificada (d^2)
        // En producción, esto usa la matriz de covarianza S invertida.
        float dx = seed.x - pred_x;
        float dy = seed.y - pred_y;
        float mahalanobis_sq = (dx*dx + dy*dy) / (sat_kalman.p + 2.0f);
        
        const float GAMMA_SQ = 9.21f; // chi-square 4 DOF, 0.99

        if (mahalanobis_sq < GAMMA_SQ && seed.confidence > 0.3f) {
            // Semilla aceptada: Actualizar Kalman con el Tracklet de la Red
            kalman_update(sat_kalman, seed.x, seed.y, seed.confidence);
        } else {
            // Semilla rechazada: Extrapolar predicción ciegamente (Coast Mode)
            sat_kalman.x = pred_x;
            sat_kalman.y = pred_y;
            sat_kalman.p += 0.1f;
        }

        // VOLCADO BINARIO (Sovereign I/O)
        if (dataset_mode) {
            int clip_idx = frame / 16;
            int frame_idx = frame % 16;
            std::string clip_dir = "RAW_DATA/clip_" + std::to_string(clip_idx);
            _mkdir(clip_dir.c_str());

            // Dump density matrix
            std::string raw_path = clip_dir + "/frame_" + std::to_string(frame_idx) + ".bin";
            std::ofstream raw_file(raw_path, std::ios::binary);
            raw_file.write((char*)d_density, width * height * sizeof(float));
            raw_file.close();

            // Dump ground truth for this frame
            std::string gt_path = clip_dir + "/gt_" + std::to_string(frame_idx) + ".txt";
            std::ofstream gt_file(gt_path);
            // format: x, y, vx, vy, confidence(1.0)
            gt_file << target_x << "," << target_y << "," << vx << "," << vy << ",1.0\n";
            gt_file.close();

            if (frame_idx == 15) {
                std::cout << "[DATASET] Clip " << clip_idx << " generado (16 frames)." << std::endl;
            }
        } else if (live_simulate) {
            // FASE 5.2: PUENTE NEURONAL A PYTHON
            // Convertir pixeles a grados simulados
            float az_deg = 180.0f + (sat_kalman.x - (width/2.0f)) * 0.1f;
            float alt_deg = 45.0f - (sat_kalman.y - (height/2.0f)) * 0.1f;
            
            telemetry_file << "{\"frame\": " << frame << ", \"az\": " << az_deg 
                           << ", \"alt\": " << alt_deg << ", \"d2\": " << mahalanobis_sq 
                           << ", \"p_det\": " << seed.confidence << "}\n";
            telemetry_file.flush();
        } else if (frame % 20 == 0) {
            std::string raw_path = "RAW_DATA/frame_" + std::to_string(frame) + ".bin";
            std::ofstream raw_file(raw_path, std::ios::binary);
            raw_file.write((char*)d_density, width * height * sizeof(float));
            raw_file.close();
            
            std::cout << "[IO] DUMP BINARIO: " << raw_path << " | SHARPNESS: " << d_v_out->sharpness << " | BEST_SO_FAR: " << best_frame << std::endl;
        }
    }

    std::cout << "[DONE] Fase 4.1 completada. Silo RAW_DATA poblado." << std::endl;
    cudaFree(d_raw_ir); cudaFree(d_density); cudaFree(d_prev); cudaFree(d_cfar); cudaFree(d_v_out);
    return 0;
}
