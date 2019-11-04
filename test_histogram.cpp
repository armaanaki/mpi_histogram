#include <cstdlib>
#include "read_file.cpp"
#include <cstdio>
#include <mpi.h>
#include <algorithm>

typedef struct histogram_struct
{
  size_t num_bins;
  double* bins;
  unsigned long long* bin_counts;
} histogram;

bool get_input(int my_rank, int comm_sz, int argc, char** argv, double*& data, unsigned long long& num_bins, unsigned long long& num_doubles);
void split_data(int my_rank, int comm_sz, double*& my_data, double*& data, unsigned long long data_size);
double find_min(double*& my_data, unsigned long long size);
double find_max(double*& my_data, unsigned long long size);
double* create_bins(double min, double max, unsigned long long num_bins);
unsigned long long* bin_counts(unsigned long long num_bins, const double* bins, const double* data, unsigned long long data_size);
unsigned long long find_bin_num (double x, const double* bins, unsigned long long num_bins);
void display_histogram (const histogram& h);

int main (int argc, char** argv) {
    // init variables
    double* data = NULL;
    double* my_data = NULL;
    double* bins;
    double max, min, my_max, my_min;
    unsigned long long num_bins, data_size;
    unsigned long long* total_bin_counts;
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

    // prep for the incoming data split
    unsigned long long my_data_size = data_size / comm_sz;
    my_data = new double[my_data_size];

    // split data
    split_data(my_rank, comm_sz, my_data, data, data_size);
    
    // find and distribute max
    my_max = find_max(my_data, my_data_size);
    my_min = find_min(my_data, my_data_size);

    MPI_Allreduce(&my_max, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&my_min, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    // create bins
    bins = create_bins(min, max, num_bins);

    // get local bin counts
    unsigned long long* my_bin_counts = bin_counts(num_bins, bins, my_data, my_data_size);

    // reduce bin counts
    if (my_rank == 0) total_bin_counts = new unsigned long long[num_bins];
    for (int i = 0; i < num_bins; i++) MPI_Reduce(&my_bin_counts[i], &total_bin_counts[i], 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // create and print histogram
    if (my_rank == 0) {
        histogram* h = new histogram;
        h->num_bins = num_bins;
        h->bin_counts = total_bin_counts;
        h->bins = bins;
        display_histogram(*h);
    }

    MPI_Finalize();
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

double find_max(double*& my_data, unsigned long long size) {
    double current_max = my_data[0];
    for (int i = 1; i < size; i++)
        if (my_data[i] > current_max) current_max = my_data[i];

    return current_max;
}

double find_min(double*& my_data, unsigned long long size) {
    double current_min = my_data[0];
    for (int i = 1; i < size; i++)
        if (my_data[i] < current_min) current_min = my_data[i];

    return current_min;
}

double* create_bins(double min, double max, unsigned long long num_bins) {
    double* bins = new double[num_bins];
    double length = (max - min)/num_bins;
    double current = min + length;
    for (size_t i = 0; i < num_bins; i++) {
        bins[i] = current;
        current += length;
    }
    return bins;
}


unsigned long long* bin_counts(unsigned long long num_bins, const double* bins, const double* data, unsigned long long data_size) {
    unsigned long long* bin_counts = new unsigned long long[num_bins];
    std::fill (bin_counts, bin_counts + num_bins, 0);
    for (int i = 0; i < data_size; i++) {
        unsigned long long bin_num = find_bin_num(data[i], bins, num_bins);
        bin_counts[bin_num]++;
    }

    return bin_counts;
}

unsigned long long find_bin_num (double x, const double* bins, unsigned long long num_bins) {
    unsigned long long i = 0;
    while (i < num_bins && x > bins[i]) i++;
    if (i == num_bins)
        i = num_bins-1; // handle round off error
    return i;
}

void display_histogram (const histogram& h) {
    double length = h.bins[1] - h.bins[0];
    double initial_left = h.bins[0] - length;
    printf ("%lf - %lf: %llu\n", initial_left, h.bins[0], h.bin_counts[0]);
    for (size_t i = 1; i < h.num_bins; i++)
        printf ("%lf - %lf: %llu\n", h.bins[i-1], h.bins[i], h.bin_counts[i]);
}
