#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <mpi.h>

long long localSum = 0;
long long globalSum = 0;

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

long long runMpi(int N, int K, int rank, int size) {
    long long localSum = 0;
    int limit = N / K;

    for (int i = rank + 1; i <= limit; i += size) {
        localSum += waysToSum(N - i, K - 1, i);
    }

    return localSum;
}

bool checkArguments(int argc, char* argv[], int& N, int& K, int rank) {
    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Ошибка: Неверное количество аргументов." << std::endl;
            std::cerr << "Использование: mpirun -n <procs> " << argv[0] << " <N> <K>" << std::endl;
        }
        return false;
    }
    try {
        N = std::stoi(argv[1]);
        K = std::stoi(argv[2]);
    }
    catch (...) {
        if (rank == 0) std::cerr << "Ошибка: Аргументы должны быть целыми числами." << std::endl;
        return false;
    }

    if (N <= 0 || K <= 0) {
        if (rank == 0) std::cerr << "Ошибка: N и K должны быть больше 0." << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {	
    MPI_Init(&argc, &argv);
	
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N, K;

    if (!checkArguments(argc, argv, N, K, rank)) {
        MPI_Finalize();
        return 1;
    }

    double seqTime = 0.0;
    double mpiTime = 0.0;
    long long seqResult = 0;

    if (rank == 0) {
        double tStart = MPI_Wtime();
        seqResult = runSequential(N, K);
        double tEnd = MPI_Wtime();
        seqTime = (tEnd - tStart) * 1000.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double tStartMpi = MPI_Wtime();

    localSum = runMpi(N, K, rank, size);

    double tEndMpi = MPI_Wtime();
    mpiTime = (tEndMpi - tStartMpi) * 1000.0;

    MPI_Reduce(&localSum, &globalSum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Параметры: N=" << N << ", K=" << K << ", Процессов=" << size << std::endl;

        std::cout << "\n1. Последовательный вариант:" << std::endl;
        std::cout << "Результат: " << seqResult << std::endl;
        std::cout << "Время:   " << std::fixed << std::setprecision(4) << seqTime << " мс" << std::endl;

        std::cout << "\n2. Параллельный вариант (MPI):" << std::endl;
        std::cout << "Результат: " << globalSum << std::endl;
        std::cout << "Время:   " << std::fixed << std::setprecision(4) << mpiTime << " мс" << std::endl;

        if (seqResult != globalSum) {
            std::cerr << "\n!!! ОШИБКА: Результаты не совпадают! !!!" << std::endl;
            std::cerr << "Seq: " << seqResult << " vs MPI: " << globalSum << std::endl;
            MPI_Finalize();
            return 1;
        }

        if (seqTime > 0 && mpiTime > 0) {
            double speedup = seqTime / mpiTime;
            double efficiency = (speedup / size) * 100.0;
            std::cout << "\n3. Ускорение (Speedup): " << std::fixed << std::setprecision(4) << speedup << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}