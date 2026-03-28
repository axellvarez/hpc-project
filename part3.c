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

// C's built-in comparison function for the final safe merge
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
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

    int *global_array = NULL;
    int *local_array = NULL;
    int chunk = N / size; 

    local_array = (int*)malloc(chunk * sizeof(int));

    if (rank == 0) {
        global_array = (int*)malloc(N * sizeof(int));
        
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

    MPI_Barrier(MPI_COMM_WORLD); 
    double start_time = MPI_Wtime();

    MPI_Scatter(global_array, chunk, MPI_INT, local_array, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process uses the Professor's Quicksort on its local unsorted chunk
    quickSort(local_array, 0, chunk - 1);

    MPI_Gather(local_array, chunk, MPI_INT, global_array, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 completes the final merge safely avoiding the O(N^2) trap
    if (rank == 0) {
        qsort(global_array, N, sizeof(int), compare);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        // Convert the decimal seconds into nanoseconds
        unsigned long long diff_ns = (unsigned long long)((end_time - start_time) * 1000000000.0);
        
        printf("Parallel Quicksort Complete\n");
        printf("elapsed time = %llu nanoseconds\n\n", diff_ns);

        printf("Starting verification...\n");
        int passed = 1;
        for (int i = 0; i < N - 1; i++) {
            if (global_array[i] > global_array[i+1]) {
                passed = 0;
                printf("Mismatch: Index %d (%d) is greater than Index %d (%d)\n", i, global_array[i], i+1, global_array[i+1]);
                break; 
            }
        }

        if (passed) {
            printf("Verification SUCCESSFUL: Array is perfectly sorted.\n");
        } else {
            printf("Verification FAILED.\n");
        }

        free(global_array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}