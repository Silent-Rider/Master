#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__        \
                      << " — " << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__host__ __device__
long long waysToSum(int target, int partsLeft, int minValue) {
    if (partsLeft == 1) {
        return (target >= minValue) ? 1LL : 0LL;
    }

    long long total = 0;
    int limit = target / partsLeft;

    for (int i = minValue; i <= limit; ++i) {
        total += waysToSum(target - i, partsLeft - 1, i);
    }
    return total;
}

__global__
void computeWaysKernel(int N, int K, unsigned long long* d_total) {
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int limit = N / K;

    if (idx < limit) {
        int i = idx + 1;
        unsigned long long ways = (unsigned long long)waysToSum(N - i, K - 1, i);
        atomicAdd(d_total, ways);
    }
}

long long runSequential(int N, int K) {
    return waysToSum(N, K, 1);
}

long long runCuda(int N, int K, int blockSize) {
    int limit = N / K;
    if (limit <= 0) return 0LL;

    unsigned long long* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));

    int gridSize = (limit + blockSize - 1) / blockSize;
    computeWaysKernel<<<gridSize, blockSize>>>(N, K, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_total = 0;
    CUDA_CHECK(cudaMemcpy(&h_total, d_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_total));

    return (long long)h_total;
}

bool parseArguments(int argc, char* argv[], int& N, int& K, int& blockSize) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Ошибка: Неверное количество аргументов." << std::endl;
        std::cerr << "Использование: " << argv[0] << " <N> <K> [число_потоков_в_блоке_CUDA]" << std::endl;
        return false;
    }

    try {
        N = atoi(argv[1]);
        K = atoi(argv[2]);
        blockSize = (argc == 4) ? atoi(argv[3]) : 256;
    }
    catch (...) {
        std::cerr << "Ошибка: все аргументы должны быть целыми числами.\n";
        return false;
    }

    if (N <= 0 || K <= 0) {
        std::cerr << "Ошибка: N и K должны быть > 0.\n";
        return false;
    }
    if (blockSize <= 0 || blockSize > 1024) {
        std::cerr << "Ошибка: blockSize должен быть в диапазоне [1, 1024].\n";
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    
    int N, K, blockSize;
    if (!parseArguments(argc, argv, N, K, blockSize)) {
        return 1;
    }

    int limit = N / K;
    int gridSize = (limit + blockSize - 1) / blockSize;

    std::cout << "Параметры: N=" << N << ", K=" << K << "" << std::endl;
    std::cout << "CUDA конфигурация: gridSize=" << gridSize << ", blockSize=" << blockSize
              << " (потоков всего: " << gridSize * blockSize << ")" << std::endl;

    std::cout << "\n1. Последовательный расчёт (CPU)" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    long long seqResult = runSequential(N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seqMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\tРезультат: " << seqResult << ", Время: " << std::fixed 
        << std::setprecision(4) << seqMs << " мс" << std::endl;


    std::cout << "\n2. Параллельный расчёт (GPU / CUDA)" << std::endl;

    runCuda(N, K, blockSize);

    auto t2 = std::chrono::high_resolution_clock::now();
    long long cudaResult = runCuda(N, K, blockSize);
    auto t3 = std::chrono::high_resolution_clock::now();
    double cudaMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "\tРезультат: " << cudaResult << ", Время: " << std::fixed 
        << std::setprecision(4) << cudaMs << " мс" << std::endl;

    if (seqResult != cudaResult) {
        std::cerr << "ОШИБКА: результаты не совпадают!" << std::endl;
        std::cerr << "CPU=" << seqResult << ", GPU=" << cudaResult << "\n";
        return 1;
    }

    if (seqMs > 0.0 && cudaMs > 0.0) {
        double speedup = seqMs / cudaMs;
        std::cout << "\nУскорение: " << std::fixed << std::setprecision(4) << speedup << std::endl;
    }
    return 0;
}
