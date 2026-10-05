// algoritmo2.cpp - Multiplicación de matrices Dividir y Conquistar (OpenMP).
// Uso: ./algoritmo2 [N n p]   (N por defecto 2048, n por defecto 256, p por defecto 1)

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>

// Vista de un bloque s x s dentro de una matriz N x N (row-major), sin copiar.
struct MatView {
    double* data;
    int stride;
    int size;
    double& at(int i, int j) const { return data[i * stride + j]; }
    MatView sub(int r, int c, int s) const { return { data + r * stride + c, stride, s }; }
};

// Valor determinista en [-1, 1] según la posición (reproducible)
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

// Hoja: multiplicación densa secuencial que acumula en C
void multiply_leaf(const MatView& A, const MatView& B, const MatView& C) {
    int s = C.size;
    for (int i = 0; i < s; ++i) {
        for (int k = 0; k < s; ++k) {
            double a = A.at(i, k);
            for (int j = 0; j < s; ++j) {
                C.at(i, j) += a * B.at(k, j);
            }
        }
    }
}

// División recursiva en cuadrantes con paralelismo de tareas
// void multiply_rec_omp(const MatView& A, const MatView& B, const MatView& C, int n) {
//     int s = C.size;
    
//     // Condición de parada (Hoja): resolver en secuencial
//     if (s <= n) {
//         multiply_leaf(A, B, C);
//         return;
//     }

//     int h = s / 2;

//     // Subproductos independientes para las 4 esquinas de C:
//     // C11 = A11*B11 + A12*B21
//     #pragma omp task
//     multiply_rec_omp(A.sub(0, 0, h), B.sub(0, 0, h), C.sub(0, 0, h), n);
//     #pragma omp task
//     multiply_rec_omp(A.sub(0, h, h), B.sub(h, 0, h), C.sub(0, 0, h), n);

//     // C12 = A11*B12 + A12*B22
//     #pragma omp task
//     multiply_rec_omp(A.sub(0, 0, h), B.sub(0, h, h), C.sub(0, h, h), n);
//     #pragma omp task
//     multiply_rec_omp(A.sub(0, h, h), B.sub(h, h, h), C.sub(0, h, h), n);

//     // C21 = A21*B11 + A22*B21
//     #pragma omp task
//     multiply_rec_omp(A.sub(h, 0, h), B.sub(0, 0, h), C.sub(h, 0, h), n);
//     #pragma omp task
//     multiply_rec_omp(A.sub(h, h, h), B.sub(h, 0, h), C.sub(h, 0, h), n);

//     // C22 = A21*B12 + A22*B22
//     #pragma omp task
//     multiply_rec_omp(A.sub(h, 0, h), B.sub(0, h, h), C.sub(h, h, h), n);
//     #pragma omp task
//     multiply_rec_omp(A.sub(h, h, h), B.sub(h, h, h), C.sub(h, h, h), n);

//     // Esperar a que todas las sub-tareas de este nivel terminen
//     #pragma omp taskwait
// }

void multiply_rec_omp(const MatView& A, const MatView& B, const MatView& C, int n) {
    int s = C.size;

    if (s <= n) {
        multiply_leaf(A, B, C);
        return;
    }

    int h = s / 2;

    MatView A11 = A.sub(0, 0, h);
    MatView A12 = A.sub(0, h, h);
    MatView A21 = A.sub(h, 0, h);
    MatView A22 = A.sub(h, h, h);

    MatView B11 = B.sub(0, 0, h);
    MatView B12 = B.sub(0, h, h);
    MatView B21 = B.sub(h, 0, h);
    MatView B22 = B.sub(h, h, h);

    MatView C11 = C.sub(0, 0, h);
    MatView C12 = C.sub(0, h, h);
    MatView C21 = C.sub(h, 0, h);
    MatView C22 = C.sub(h, h, h);

    #pragma omp task
    {
        multiply_rec_omp(A11, B11, C11, n);
        multiply_rec_omp(A12, B21, C11, n);
    }

    #pragma omp task
    {
        multiply_rec_omp(A11, B12, C12, n);
        multiply_rec_omp(A12, B22, C12, n);
    }

    #pragma omp task
    {
        multiply_rec_omp(A21, B11, C21, n);
        multiply_rec_omp(A22, B21, C21, n);
    }

    #pragma omp task
    {
        multiply_rec_omp(A21, B12, C22, n);
        multiply_rec_omp(A22, B22, C22, n);
    }

    #pragma omp taskwait
}

int main(int argc, char** argv) {
    // Lectura de parámetros desde la terminal
    int N = (argc > 1) ? std::atoi(argv[1]) : 2048; // Tamaño matriz
    int n = (argc > 2) ? std::atoi(argv[2]) : 256;  // Tamaño bloque
    int p = (argc > 3) ? std::atoi(argv[3]) : 1;    // Threads

    omp_set_num_threads(p);

    std::vector<double> A(N * N), B(N * N), C(N * N, 0.0);
    init_matrix(A, N, 1);
    init_matrix(B, N, 2);

    MatView Av{ A.data(), N, N }, Bv{ B.data(), N, N }, Cv{ C.data(), N, N };

    // auto t0 = std::chrono::high_resolution_clock::now();
    auto t0 = std::chrono::steady_clock::now();
    
    // Crear la región paralela e invocar la primera tarea desde un único hilo (single)
    #pragma omp parallel
    {
        #pragma omp single
        {
            multiply_rec_omp(Av, Bv, Cv, n);
        }
    }

    // auto t1 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double checksum = 0.0;
    for (double v : C) {
        checksum += v;
    }

    std::printf("N=%d n=%d threads=%d tiempo=%.2f ms checksum=%.6f\n", N, n, p, ms, checksum);
    return 0;
}