#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-9

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default matrix dimensions. Can be overridden via command line arguments.
    int M = 512, N = 512, K = 512;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    } else if (rank == 0) {
        printf("Usage: mpirun -np <procs> ./part2b M N K\n");
        printf("Using default dimensions: 512 x 512 x 512\n\n");
    }

    // Pointers for the full matrices (only allocated on Rank 0)
    double *A = NULL, *C = NULL;
    
    // Pointers for the local chunks every process needs
    double *B = (double*)malloc(N * K * sizeof(double));
    double *A_local = NULL;
    double *C_local = NULL;

    // Calculate block sizes. 
    // For MPI_Scatter/Gather, we assume M is perfectly divisible by size.
    int chunk = M / size; 

    A_local = (double*)malloc(chunk * N * sizeof(double));
    C_local = (double*)malloc(chunk * K * sizeof(double));

    if (rank == 0) {
        // Allocate and initialize full matrices A and C
        A = (double*)malloc(M * N * sizeof(double));
        C = (double*)malloc(M * K * sizeof(double));
        
        srand(time(NULL));
        for (int i = 0; i < M * N; i++) A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < N * K; i++) B[i] = (double)rand() / RAND_MAX;

        printf("Starting Collective Matrix Multiplication for size %d x %d x %d...\n", M, N, K);
    }

    // Sync all processes before starting the timer to ensure accurate timing
    MPI_Barrier(MPI_COMM_WORLD); 
    
    // Start the parallel execution timer
    double start_time = MPI_Wtime();

    // Broadcast full matrix B to all processes
    MPI_Bcast(B, N * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distribute equal blocks of A to all processes (including Rank 0)
    MPI_Scatter(A, chunk * N, MPI_DOUBLE, A_local, chunk * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local matrix block C_local
    for (int i = 0; i < chunk; i++) {
        for (int j = 0; j < K; j++) {
            double sum = 0.0;
            for (int l = 0; l < N; l++) {
                sum += A_local[i * N + l] * B[l * K + j];
            }
            C_local[i * K + j] = sum;
        }
    }

    // Aggregate results into the final matrix C using Gather
    MPI_Gather(C_local, chunk * K, MPI_DOUBLE, C, chunk * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Stop the parallel execution timer
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Parallel Matrix Multiplication Complete\n");
        printf("Elapsed time: %f seconds\n\n", end_time - start_time);

        // Verification (Not included in parallel timing)
        printf("Starting verification against sequential calculation...\n");
        double *C_seq = (double*)malloc(M * K * sizeof(double));
        
        // Sequential matrix multiplication
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                double sum = 0.0;
                for (int l = 0; l < N; l++) {
                    sum += A[i * N + l] * B[l * K + j];
                }
                C_seq[i * K + j] = sum;
            }
        }

        // Compare matrices using the 10^-9 threshold
        int passed = 1;
        for (int i = 0; i < M * K; i++) {
            if (fabs(C_seq[i] - C[i]) > EPSILON) {
                passed = 0;
                printf("Mismatch at index %d: Seq=%f, Par=%f\n", i, C_seq[i], C[i]);
                break; // Stop at the first error to avoid flooding the terminal
            }
        }

        if (passed) {
            printf("Verification SUCCESSFUL: |a - b| < 10^-9 threshold met.\n");
        } else {
            printf("Verification FAILED.\n");
        }

        free(C_seq);
        free(A);
        free(C);
    }

    // Clean up local memory
    free(A_local);
    free(B);
    free(C_local);

    MPI_Finalize();
    return 0;
}