#include <cstdlib>
#include "read_file.cpp"
#include <cstdio>
#include <mpi.h>

bool get_input(int my_rank, int comm_sz, int argc, char** argv, double*& data, unsigned long long& num_bins, unsigned long long& num_doubles);
void split_data(int my_rank, int comm_sz, double*& my_data, double*& data, unsigned long long data_size);

typedef struct histogram_struct
{
  size_t num_bins;
  double* bins;
  unsigned long long* bin_counts;
} histogram;

int main (int argc, char** argv) {
    // init variables
    histogram* h = NULL;
    double* data = NULL;
    double* my_data = NULL;
    unsigned long long num_bins, data_size;
    int my_rank, comm_sz;
    
    // init MPI variables
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // get file input and distribute input, if there is an error then all will exit
    bool error = false;
    if (my_rank == 0)
        error = get_input(my_rank, comm_sz, argc, argv, data, num_bins, data_size); 
    
    if (error) {
        MPI_Finalize();
        exit(-1);
    }

    MPI_Bcast(&num_bins, sizeof(num_bins), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&error, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data_size, sizeof(data_size), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    my_data = new double[data_size / comm_sz];

    split_data(my_rank, comm_sz, my_data, data, data_size);
}

bool get_input(int my_rank, int comm_sz, int argc, char** argv, double*& data, unsigned long long& num_bins, unsigned long long& num_doubles) {
    if (argc != 3) {
        printf("expected <file_name> <number of intervals>\n");
        return true;
    } else {
        unsigned long long num_bytes = get_file_size(argv[1]);
        num_doubles = num_bytes/sizeof(double);
        data = read_from_file (argv[1], num_doubles);
        num_bins = strtol(argv[2], NULL, 10);
        if (num_doubles % comm_sz != 0) {
            printf("Please make sure the data is divisible by the total processes\n");
            return true;
        }
    }

    return false;
}

void split_data(int my_rank, int comm_sz, double*& my_data, double*& data, unsigned long long data_size) {
    unsigned long long size_per_process = data_size/comm_sz;

    if (my_rank == 0) {
        for (int i = 0; i < comm_sz; i++) {
            double* array = new double[size_per_process];

            for (int j = 0; j < size_per_process; j++)
                if (i != 0) array[j] = data[j + (size_per_process * i)];
                else my_data[j] = data[j];
            

            if (i != 0) MPI_Send(array, size_per_process, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else MPI_Recv(my_data, size_per_process, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
