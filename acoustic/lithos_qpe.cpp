#include <cmath>
#include <complex>

double estimate_quantum_phase(float* signal_buffer, int buffer_size, double reference_freq, double sample_rate) {
    std::complex<double> accumulated_phase(0, 0);
    
    for (int n = 0; n < buffer_size; ++n) {
        // Generación de referencia local
        double ref_angle = 2.0 * M_PI * reference_freq * (double)n / sample_rate;
        std::complex<double> ref_vector(cos(ref_angle), sin(ref_angle));
        
        // Producto interno para detectar rotación de fase
        accumulated_phase += (double)signal_buffer[n] * ref_vector;
    }
    
    // Retorna el ángulo phi inducido por la perturbación
    return std::arg(accumulated_phase);
}
