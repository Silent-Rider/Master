#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <omp.h>

long long seqResult = 0;
long long ompResult = 0;

long long waysToSum(int target, int partsLeft, int minValue) {
    if (partsLeft == 1) {
        return (target >= minValue) ? 1 : 0;
    }

    long long totalWays = 0;
    int limit = target / partsLeft;

    for (int i = minValue; i <= limit; ++i) {
        totalWays += waysToSum(target - i, partsLeft - 1, i);
    }
    return totalWays;
}

long long runSequential(int N, int K) {
    return waysToSum(N, K, 1);
}

long long runParallel(int N, int K, int numThreads) {
    long long totalWays = 0;
    int limit = N / K;
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) reduction(+:totalWays)
    for (int i = 1; i <= limit; ++i) {
        totalWays += waysToSum(N - i, K - 1, i);
    }
    return totalWays;
}

bool checkArguments(int argc, char* argv[], int& N, int& K, int& numThreads) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Ошибка: Неверное количество аргументов." << std::endl;
        std::cerr << "Использование: " << argv[0] << " <N> <K> [число_потоков]" << std::endl;
        return false;
    }
    try {
        N = std::stoi(argv[1]);
        K = std::stoi(argv[2]);
        numThreads = (argc == 4) ? std::stoi(argv[3]) : omp_get_max_threads();
    }
    catch (const std::invalid_argument&) {
        std::cerr << "Ошибка: Аргументы должны быть целыми числами." << std::endl;
        return false;
    }
    catch (const std::out_of_range&) {
        std::cerr << "Ошибка: Числа выходят за допустимый диапазон." << std::endl;
        return false;
    }
    struct { int val; const char* name; } checks[] = {
        {N, "N"}, {K, "K"}, {numThreads, "количество потоков"}
    };
    for (auto& [val, name] : checks) {
        if (val <= 0) {
            std::cerr << "Ошибка: " << name << " должно быть больше 0.\n";
            return false;
        }
    }
    return true;
}

bool verifyResults(long long seq, long long omp) {
    if (seq != omp) {
        std::cerr << "ОШИБКА: Результаты не совпадают!" << std::endl;
        std::cerr << "Последовательный вариант: " << seq << ", Параллельный вариант: " << omp << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    int N, K, numThreads;

    if (!checkArguments(argc, argv, N, K, numThreads)) {
        return 1;
    }
    double seqTimeMs = 0.0;
    double ompTimeMs = 0.0;

    auto startSeq = std::chrono::high_resolution_clock::now();
    seqResult = runSequential(N, K);
    auto endSeq = std::chrono::high_resolution_clock::now();
    seqTimeMs = std::chrono::duration<double, std::milli>(endSeq - startSeq).count();

    ompResult = 0;
    auto startOmp = std::chrono::high_resolution_clock::now();
    ompResult = runParallel(N, K, numThreads);
    auto endOmp = std::chrono::high_resolution_clock::now();
    ompTimeMs = std::chrono::duration<double, std::milli>(endOmp - startOmp).count();

    std::cout << "Параметры: N=" << N << ", K=" << K << ", Потоков=" << numThreads << std::endl;
    std::cout << "Последовательно: Результат=" << seqResult
        << ", Время=" << std::fixed << std::setprecision(4) << seqTimeMs << " мс" << std::endl;
    std::cout << "Параллельно (OpenMP): Результат=" << ompResult
        << ", Время=" << std::fixed << std::setprecision(4) << ompTimeMs << " мс" << std::endl;

    if (!verifyResults(seqResult, ompResult)) {
        return 1;
    }
    if (seqTimeMs > 0 && ompTimeMs > 0) {
        double speedup = seqTimeMs / ompTimeMs;
        std::cout << "Ускорение: " << std::fixed << std::setprecision(4) << speedup << std::endl;
    }
    return 0;
}