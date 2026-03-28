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
        printf("Usage: mpirun -np <procs> ./part1 M N K\n");
        printf("Using default dimensions: 512 x 512 x 512\n\n");
    }

    // Pointers for the full matrices (only allocated on Rank 0)
    double *A = NULL, *C = NULL;
    
    // Pointers for the local chunks every process needs
    double *B = (double*)malloc(N * K * sizeof(double));
    double *A_local = NULL;
    double *C_local = NULL;

    // Calculate block sizes. 
    // Workers (Ranks 1 to size-1) get standard chunks.
    // Main thread (Rank 0) takes the last block, which includes any remainder.
    int chunk = M / size;
    int local_rows;
    if (rank == 0) {
        local_rows = M - (size - 1) * chunk; // The last block + remainder
    } else {
        local_rows = chunk;
    }

    A_local = (double*)malloc(local_rows * N * sizeof(double));
    C_local = (double*)malloc(local_rows * K * sizeof(double));

    if (rank == 0) {
        // Allocate and initialize full matrices A and C
        A = (double*)malloc(M * N * sizeof(double));
        C = (double*)malloc(M * K * sizeof(double));
        
        srand(time(NULL));
        for (int i = 0; i < M * N; i++) A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < N * K; i++) B[i] = (double)rand() / RAND_MAX;

        printf("Starting Point-to-Point Matrix Multiplication for size %d x %d x %d...\n", M, N, K);

        // Start the parallel execution timer
        double start_time = MPI_Wtime();

        // Send full matrix B to all worker processes
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(B, N * K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }

        // Distribute blocks of A to worker processes
        for (int dest = 1; dest < size; dest++) {
            int offset = (dest - 1) * chunk * N;
            MPI_Send(&A[offset], chunk * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
        }

        // Copy the last block into Rank 0's local memory
        int offset_rank0 = (size - 1) * chunk * N;
        for (int i = 0; i < local_rows * N; i++) {
            A_local[i] = A[offset_rank0 + i];
        }

        // Rank 0 computes its own block (the last block)
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < K; j++) {
                double sum = 0.0;
                for (int l = 0; l < N; l++) {
                    sum += A_local[i * N + l] * B[l * K + j];
                }
                C_local[i * K + j] = sum;
            }
        }

        // Aggregate results into the final matrix C
        // Copy Rank 0's computed block into the correct position in C
        int c_offset_rank0 = (size - 1) * chunk * K;
        for (int i = 0; i < local_rows * K; i++) {
            C[c_offset_rank0 + i] = C_local[i];
        }

        // Receive the computed blocks from all worker processes
        for (int source = 1; source < size; source++) {
            int c_offset = (source - 1) * chunk * K;
            MPI_Recv(&C[c_offset], chunk * K, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Stop the parallel execution timer
        double end_time = MPI_Wtime();
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

    } else {
        
        // WORKER PROCESS LOGIC
        // Receive matrix B and local chunk of A
        MPI_Recv(B, N * K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(A_local, local_rows * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute local matrix block C_local
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < K; j++) {
                double sum = 0.0;
                for (int l = 0; l < N; l++) {
                    sum += A_local[i * N + l] * B[l * K + j];
                }
                C_local[i * K + j] = sum;
            }
        }

        // Send the computed block back to Rank 0
        MPI_Send(C_local, local_rows * K, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    // Clean up local memory
    free(A_local);
    free(B);
    free(C_local);

    MPI_Finalize();
    return 0;
}