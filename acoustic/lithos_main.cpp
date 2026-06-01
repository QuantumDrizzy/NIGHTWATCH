#include <iostream>
#include <portaudio.h>
#include <cmath>

extern double estimate_quantum_phase(float* signal_buffer, int buffer_size, double reference_freq, double sample_rate);

static int lithos_callback(const void *inputBuffer, void *outputBuffer, unsigned long framesPerBuffer, 
                          const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void *userData) {
    float *in = (float*)inputBuffer;
    if (!in) return paContinue;

    // Estimación a 1kHz de frecuencia de referencia
    double phi = estimate_quantum_phase(in, framesPerBuffer, 1000.0, 96000.0);
    
    // Intensidad RMS
    float rms = 0;
    for(int i=0; i<framesPerBuffer; i++) rms += in[i]*in[i];
    rms = sqrt(rms/framesPerBuffer);

    printf("\r[LITHOS] Phase Shift: %.4f rad | Intensity: %.4f", phi, rms);
    fflush(stdout);
    
    return paContinue;
}

int main() {
    PaError err = Pa_Initialize();
    if (err != paNoError) return 1;

    PaStream *stream;
    err = Pa_OpenDefaultStream(&stream, 1, 0, paFloat32, 96000, 512, lithos_callback, NULL);
    if (err != paNoError) return 1;

    err = Pa_StartStream(stream);
    if (err != paNoError) return 1;

    std::cout << "[*] LITHOS-SCAN Activo (96kHz/24-bit). Pulsa ENTER para detener." << std::endl;
    std::cin.get();

    Pa_StopStream(stream);
    Pa_Terminate();

    return 0;
}
