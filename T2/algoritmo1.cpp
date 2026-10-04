// algoritmo1.cpp - Multiplicación de matrices con Paralelismo de Datos (OpenMP).
// Uso: ./algoritmo1 [N p]   (N por defecto 2048, p por defecto 1 thread)

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>

// Valor determinista en [-1, 1] segun la posicion (reproducible)
double det_value(int i, int j, double offset) {
    return std::sin(i * 0.1 + j * 0.3 + offset);
}

void init_matrix(std::vector<double>& M, int N, double offset) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            M[i * N + j] = det_value(i, j, offset);
        }
    }
}

// Algoritmo 1: Paralelismo de datos con 3 loops anidados (i, k, j)
void multiply_data_parallel(const std::vector<double>& A, 
                            const std::vector<double>& B, 
                            std::vector<double>& C, 
                            int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    // 1. Leer parametros desde la terminal
    int N = (argc > 1) ? std::atoi(argv[1]) : 2048; // Tamaño de la matriz
    int p = (argc > 2) ? std::atoi(argv[2]) : 1;    // Número de threads

    // 2. Configurar el número de threads en OpenMP
    omp_set_num_threads(p);

    // 3. Crear e inicializar matrices
    std::vector<double> A(N * N), B(N * N), C(N * N, 0.0);
    init_matrix(A, N, 1);
    init_matrix(B, N, 2);

    // 4. Medir tiempo de ejecución de la multiplicación paralela
    auto t0 = std::chrono::high_resolution_clock::now();
    multiply_data_parallel(A, B, C, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 5. Calcular checksum para validación de corrección
    double checksum = 0.0;
    for (double v : C) {
        checksum += v;
    }

    // 6. Imprimir resultados en pantalla
    std::printf("N=%d threads=%d tiempo=%.2f ms checksum=%.6f\n", N, p, ms, checksum);

    return 0;
}