#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ======================================================================
// PROFESSOR'S SEQUENTIAL QUICKSORT FUNCTIONS
// ======================================================================
void swap(int* a, int* b) {
    int t = *a; *a = *b; *b = t;
}

int partition (int arr[], int low, int high) {
    int pivot = arr[high];      // pivot
    int i = (low - 1);          // Index of smaller element

    for (int j = low; j <= high- 1; j++) {
        if (arr[j] < pivot) {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
// ======================================================================

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default array size. Can be overridden via command line arguments.
    int N = 1000000; 
    if (argc == 2) {
        N = atoi(argv[1]);
    } else if (rank == 0) {
        printf("Usage: mpirun -np <procs> ./part3 N\n");
        printf("Using default array size: 1,000,000\n\n");
    }

    // Pointer for the full array (only allocated on Rank 0)
    int *global_array = NULL;
    
    // Pointer for the local chunks every process needs
    int *local_array = NULL;

    // Calculate block sizes. 
    // For MPI_Scatter/Gather, we assume N is perfectly divisible by size.
    int chunk = N / size; 

    local_array = (int*)malloc(chunk * sizeof(int));

    if (rank == 0) {
        // Allocate the full array
        global_array = (int*)malloc(N * sizeof(int));
        
        // Read the unsorted array using the professor's exact file reading logic
        char str[100];
        int count = 0;
        FILE* fp = fopen("data.txt", "r");
        if (fp == NULL) {
            printf("CRITICAL ERROR: Could not open data.txt.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        while (fscanf(fp, "%s", str) != EOF && count < N) {
            global_array[count] = atoi(str);
            count++;
        }
        fclose(fp);
        
        if (count != N) {
            printf("Warning: Expected %d elements but read %d.\n", N, count);
        }

        printf("Starting Parallel Quicksort for %d elements...\n", N);
    }

    // Sync all processes before starting the timer to ensure accurate timing
    MPI_Barrier(MPI_COMM_WORLD); 
    
    // Start the parallel execution timer
    double start_time = MPI_Wtime();

    // Distribute equal blocks of the array to all processes (including Rank 0)
    MPI_Scatter(global_array, chunk, MPI_INT, local_array, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local sorting block 
    // Each process uses the Professor's Quicksort on its local chunk
    quickSort(local_array, 0, chunk - 1);

    // Aggregate results into the final array using Gather
    MPI_Gather(local_array, chunk, MPI_INT, global_array, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 completes the final merge by doing one last quicksort pass
    if (rank == 0) {
        quickSort(global_array, 0, N - 1);
    }

    // Stop the parallel execution timer
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Parallel Quicksort Complete\n");
        printf("Elapsed time: %f seconds\n\n", end_time - start_time);

        // Verification (Not included in parallel timing)
        printf("Starting verification...\n");
        
        // Compare elements to ensure ascending order
        int passed = 1;
        for (int i = 0; i < N - 1; i++) {
            if (global_array[i] > global_array[i+1]) {
                passed = 0;
                printf("Mismatch: Index %d (%d) is greater than Index %d (%d)\n", i, global_array[i], i+1, global_array[i+1]);
                break; // Stop at the first error to avoid flooding the terminal
            }
        }

        if (passed) {
            printf("Verification SUCCESSFUL: Array is perfectly sorted.\n");
        } else {
            printf("Verification FAILED.\n");
        }

        free(global_array);
    }

    // Clean up local memory
    free(local_array);

    MPI_Finalize();
    return 0;
}