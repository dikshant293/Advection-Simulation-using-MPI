#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// helper function to send all the square partitions to rank 0 for combining and writing to output file
void send_recv_data_and_write(double **C_n, int mype, int N, int x_chunk_size, int y_chunk_size, int nprocs, MPI_Status *stat, int x_n_chunks, int y_n_chunks, FILE *file_start_end)
{   
    // arrays for recieving data from non zero ranks
    double **C, *C_recv;
    if (mype == 0)
    {
        // C is the NxN matrix that is written to the output file
        C = malloc(N * sizeof(double *));
        C[0] = malloc(N * N * sizeof(double));
        for (int i = 0; i < N; i++)
        {
            C[i] = C[0] + i * N;
        }
        // 1D continuous array to receive other rank's C matrix
        C_recv = malloc(x_chunk_size * y_chunk_size * sizeof(double));
        
        // fill rank 0 's positions in C
        for (int i = 0; i < x_chunk_size; i++)
            for (int j = 0; j < y_chunk_size; j++)
                C[i][j] = C_n[i][j];
        
        // receive others' data and fill in appropriate places in C
        for (int from_pe = 1; from_pe < nprocs; from_pe++)
        {
            MPI_Recv(C_recv, x_chunk_size * y_chunk_size, MPI_DOUBLE, from_pe, MPI_ANY_TAG, MPI_COMM_WORLD, stat);
            int local_x_start, local_y_start;
            local_x_start = x_chunk_size * (from_pe / y_n_chunks);
            local_y_start = y_chunk_size * (from_pe % y_n_chunks);
            for (int i = 0; i < x_chunk_size * y_chunk_size; i++)
            {
                int local_i, local_j;
                local_i = local_x_start + i / y_chunk_size;
                local_j = local_y_start + i % y_chunk_size;
                if (local_i >= N || local_j >= N)
                    continue;
                C[local_i][local_j] = C_recv[i];
            }
        }
        // write to output file
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                fprintf(file_start_end, "%e ", C[i][j]);
            }
            fprintf(file_start_end, "\n");
        }
        free(C[0]);
        free(C);
    }
    else
    {
        // send data to rank 0
        MPI_Send(C_n[0], x_chunk_size * y_chunk_size, MPI_DOUBLE, 0, 69, MPI_COMM_WORLD);
    }
}

// helper function to MPI send only if the distination is different from the source, else just swap local pointers to 1D arrays
void send_helper(double **send_arr, int n, int from, int to, int tag, double **recv_arr)
{
    if (from == to)
    {
        double *temp = *send_arr;
        *send_arr = *recv_arr;
        *recv_arr = temp;
    }
    else
    {
        MPI_Send(*send_arr, n, MPI_DOUBLE, to, tag, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    int nprocs, mype;
    int x_start, x_end, y_start, y_end, x_n_chunks, y_n_chunks, x_chunk_size, y_chunk_size, x_idx, y_idx;
    int N = 1, NT = 1, nthreads = omp_get_num_procs(), method = 1;
    double L = 1.0f, T = 1.0f, u = 1.0f, v = 1.0f, del_x = 1.0f, del_t = 1.0f, t1, t2;
    MPI_Status stat;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);

    FILE *file_start_end;
    if (mype == 0)
        printf("max avail threads = %d\n", nthreads);

    if (argc < 6)
    {
        printf("Insufficient Arguments..exiting\n");
        return 0;
    }

    N = atoi(argv[1]);
    L = atof(argv[2]);
    T = atof(argv[3]);
    u = atof(argv[4]);
    v = atof(argv[5]);
    del_x = L / (N - 1);
    del_t = 0.5f * del_x / sqrt(u * u + v * v);

    if (argc > 6)
        nthreads = atoi(argv[6]);

    // id del_t is spcified in CLI, then take it, else use Courant condition value
    if (argc > 7)
        del_t = atof(argv[7]);
    
    NT = T / del_t;

    if (mype == 0)
    {
        printf("N = %d\tNT = %d\nnthreads = %d\tnprocs = %d\n", N, NT, nthreads, nprocs);
    }

    // factorize nprocs to make the partition grid as close to a perfect square as possible
    x_n_chunks = 1;
    y_n_chunks = nprocs;
    for (int i = 1; i * i <= nprocs; i++)
        if (nprocs % i == 0)
        {
            x_n_chunks = i;
            y_n_chunks = nprocs / i;
        }

    // chunk_size is the number of grid points in x and y direction for each rank
    x_chunk_size = (N + x_n_chunks - 1) / x_n_chunks;
    y_chunk_size = (N + y_n_chunks - 1) / y_n_chunks;
    // index of the rank, from 0,0 to nprocs-1,nprocs-1
    x_idx = mype / y_n_chunks;
    y_idx = mype % y_n_chunks;
    // start and end indicies in the parent grid for 0,0 in the rank chunk
    x_start = x_chunk_size * x_idx;
    y_start = y_chunk_size * y_idx;
    x_end = MIN(N, x_chunk_size * (x_idx + 1));
    y_end = MIN(N, y_chunk_size * (y_idx + 1));

    int mid = NT / 2;
    double n_cells = (double)N * (double)N * (double)NT;

    if (mype == 0)
    {
        // start writing initial general information in the output file
        file_start_end = fopen("startend.dat", "w");
        fprintf(file_start_end, "%d %d %e %e %d %d %d %d\n", N, NT, u, v, mid, 1, nthreads, nprocs);
    }

    // allocate 2D arrays for local computation of timesteps 
    double **C_n, **C_n_1, **C_n_half;
    C_n = malloc(x_chunk_size * sizeof(double *));
    C_n_1 = malloc(x_chunk_size * sizeof(double *));
    C_n_half = malloc(x_chunk_size * sizeof(double *));
    C_n[0] = malloc(x_chunk_size * y_chunk_size * sizeof(double));
    C_n_1[0] = malloc(x_chunk_size * y_chunk_size * sizeof(double));
    C_n_half[0] = malloc(x_chunk_size * y_chunk_size * sizeof(double));
    for (int i = 0; i < x_chunk_size; i++)
    {
        C_n[i] = C_n[0] + i * y_chunk_size;
        C_n_1[i] = C_n_1[0] + i * y_chunk_size;
        C_n_half[i] = C_n_half[0] + i * y_chunk_size;
    }

    // initialize the C_n array
#pragma omp parallel for num_threads(nthreads) default(none) shared(L, N, C_n, x_chunk_size, y_chunk_size, x_start, y_start, mype, nprocs) schedule(static)
    for (int i = 0; i < x_chunk_size; i++)
    {
        for (int j = 0; j < y_chunk_size; j++)
        {
            double x = -L * 0.5f + (i + x_start) * (L / N), y = -L * 0.5f + (j + y_start) * (L / N);
            C_n[i][j] = x <= 0.5 && x >= -0.5 && y <= 0.1 && y >= -0.1 ? 1.0f : 0.0f;
        }
    }

    // send to rank 0 where its written to output
    send_recv_data_and_write(C_n, mype, N, x_chunk_size, y_chunk_size, nprocs, &stat, x_n_chunks, y_n_chunks, file_start_end);
    
    // wait for all ranks to reach this barrier
    MPI_Barrier(MPI_COMM_WORLD);

    // in the square grid with numbered top to bottom row wise from 0 to nprocs-1
    // with top left square rank 0 and bottom right cell rank nprocs-1
    // the neigbours to a cell is the top (north), down (south) left (west) and right (east)
    // we calculate the ranks of the 4 neihghbours here 
    int left_pe, right_pe, up_pe, down_pe;
    left_pe = mype % y_n_chunks == 0 ? mype + y_n_chunks - 1 : mype - 1;
    right_pe = (mype + 1) % y_n_chunks == 0 ? mype - y_n_chunks + 1 : mype + 1;
    up_pe = mype / y_n_chunks == 0 ? mype + y_n_chunks * (x_n_chunks - 1) : mype - y_n_chunks;
    down_pe = mype / y_n_chunks == x_n_chunks - 1 ? mype - y_n_chunks * (x_n_chunks - 1) : mype + y_n_chunks;

    // allocate 1D arrays for sending and recieving data

    double *send_up, *send_down, *send_left, *send_right;
    send_up = malloc(y_chunk_size * sizeof(double));
    send_down = malloc(y_chunk_size * sizeof(double));
    send_left = malloc(x_chunk_size * sizeof(double));
    send_right = malloc(x_chunk_size * sizeof(double));

    double *recv_up, *recv_down, *recv_left, *recv_right;
    recv_up = malloc(y_chunk_size * sizeof(double));
    recv_down = malloc(y_chunk_size * sizeof(double));
    recv_left = malloc(x_chunk_size * sizeof(double));
    recv_right = malloc(x_chunk_size * sizeof(double));

    double x, y, u_at_pos, v_at_pos, C_i_j, C_i_j_minus_1, C_i_j_plus_1, C_i_minus_1_j, C_i_plus_1_j, sum;

    // start timer
    t1 = omp_get_wtime();
    for (int k = 0; k < NT; k++)
    {   
        // obtain the top and bottom boundaries for sending
        for (int j = 0; j < y_chunk_size; j++)
        {
            send_up[j] = C_n[0][j];
            send_down[j] = C_n[x_end - x_start - 1][j];
        }
        // obtain the left and right boundaries for sending
        for (int i = 0; i < x_chunk_size; i++)
        {
            send_left[i] = C_n[i][0];
            send_right[i] = C_n[i][y_end - y_start - 1];
        }

        // interleaved send and receive between neighbours to ensure it never deadlocks
        // first do left and right boundaries send and receive
        if (y_idx % 2 == 0)
        {   
            // send right
            send_helper(&send_right, x_chunk_size, mype, right_pe, 3, &recv_left);
            // receive right (if neighbour different from self)
            if (mype != right_pe)
                MPI_Recv(recv_right, x_chunk_size, MPI_DOUBLE, right_pe, 2, MPI_COMM_WORLD, &stat);
            // send left
            send_helper(&send_left, x_chunk_size, mype, left_pe, 2, &recv_right);
            // receive left (if neighbour different from self)
            if (mype != left_pe)
                MPI_Recv(recv_left, x_chunk_size, MPI_DOUBLE, left_pe, 3, MPI_COMM_WORLD, &stat);
        }
        else
        {   
            // receive left (if neighbour different from self)
            if (mype != left_pe)
                MPI_Recv(recv_left, x_chunk_size, MPI_DOUBLE, left_pe, 3, MPI_COMM_WORLD, &stat);
            // send left
            send_helper(&send_left, x_chunk_size, mype, left_pe, 2, &recv_right);
            // receive right (if neighbour different from self)
            if (mype != right_pe)
                MPI_Recv(recv_right, x_chunk_size, MPI_DOUBLE, right_pe, 2, MPI_COMM_WORLD, &stat);
            // send right
            send_helper(&send_right, x_chunk_size, mype, right_pe, 3, &recv_left);
        }

        if (x_idx % 2 == 0)
        {   
            // send up
            send_helper(&send_up, y_chunk_size, mype, up_pe, 0, &recv_down);
            // receive up (if neighbour different from self)
            if (mype != up_pe)
                MPI_Recv(recv_up, y_chunk_size, MPI_DOUBLE, up_pe, 1, MPI_COMM_WORLD, &stat);
            // send down
            send_helper(&send_down, y_chunk_size, mype, down_pe, 1, &recv_up);
            // receive down (if neighbour different from self)
            if (mype != down_pe)
                MPI_Recv(recv_down, y_chunk_size, MPI_DOUBLE, down_pe, 0, MPI_COMM_WORLD, &stat);
        }
        else
        {
            // receive down (if neighbour different from self)
            if (mype != down_pe)
                MPI_Recv(recv_down, y_chunk_size, MPI_DOUBLE, down_pe, 0, MPI_COMM_WORLD, &stat);
            // send down
            send_helper(&send_down, y_chunk_size, mype, down_pe, 1, &recv_up);
            // receive up (if neighbour different from self)
            if (mype != up_pe)
                MPI_Recv(recv_up, y_chunk_size, MPI_DOUBLE, up_pe, 1, MPI_COMM_WORLD, &stat);
            // send up
            send_helper(&send_up, y_chunk_size, mype, up_pe, 0, &recv_down);
        }

        // fill boundary where i is either 0 or x_chunk_size - 1 ,i.e, left and right boundary 
        for (int i = 0; i < x_chunk_size; i += x_chunk_size - 1)
        {
            for (int j = 0; j < y_chunk_size; j++)
            {
                x = -L * 0.5f + (i + x_start) * (L / N);
                y = -L * 0.5f + (j + y_start) * (L / N);
                u_at_pos = u * y;
                v_at_pos = v * x;
                C_i_j = C_n[i][j];

                C_i_minus_1_j = i == 0 ? recv_up[j] : C_n[i - 1][j];
                C_i_j_minus_1 = j == 0 ? recv_left[i] : C_n[i][j - 1];
                C_i_plus_1_j = i == x_chunk_size - 1 ? recv_down[j] : C_n[i + 1][j];
                C_i_j_plus_1 = j == y_chunk_size - 1 ? recv_right[i] : C_n[i][j + 1];

                sum = C_i_j_minus_1 + C_i_j_plus_1 + C_i_minus_1_j + C_i_plus_1_j;
                C_n_1[i][j] = 0.25f * sum - 0.5f * del_t / del_x * (u_at_pos * (C_i_plus_1_j - C_i_minus_1_j) + v_at_pos * (C_i_j_plus_1 - C_i_j_minus_1));

                if (k == mid)
                    C_n_half[i][j] = C_n_1[i][j];
            }
        }

        // fill boundary where j is either 0 or y_chunk_size - 1 ,i.e, top and down boundary
        for (int j = 0; j < y_chunk_size; j += y_chunk_size - 1)
        {
            for (int i = 0; i < x_chunk_size; i++)
            {
                x = -L * 0.5f + (i + x_start) * (L / N);
                y = -L * 0.5f + (j + y_start) * (L / N);
                u_at_pos = u * y;
                v_at_pos = v * x;
                C_i_j = C_n[i][j];

                C_i_minus_1_j = i == 0 ? recv_up[j] : C_n[i - 1][j];
                C_i_j_minus_1 = j == 0 ? recv_left[i] : C_n[i][j - 1];
                C_i_plus_1_j = i == x_chunk_size - 1 ? recv_down[j] : C_n[i + 1][j];
                C_i_j_plus_1 = j == y_chunk_size - 1 ? recv_right[i] : C_n[i][j + 1];

                sum = C_i_j_minus_1 + C_i_j_plus_1 + C_i_minus_1_j + C_i_plus_1_j;
                C_n_1[i][j] = 0.25f * sum - 0.5f * del_t / del_x * (u_at_pos * (C_i_plus_1_j - C_i_minus_1_j) + v_at_pos * (C_i_j_plus_1 - C_i_j_minus_1));

                if (k == mid)
                    C_n_half[i][j] = C_n_1[i][j];
            }
        }

        // parallaly compute all the internal points without the use of any mod operations
#pragma omp parallel for num_threads(nthreads) default(none) shared(x_start, y_start, del_t, del_x, u, v, k, NT, file_start_end, N, C_n, C_n_1, C_n_half, mid, x_chunk_size, y_chunk_size, recv_down, recv_left, recv_right, recv_up, L) private(x, y, u_at_pos, v_at_pos, C_i_j, C_i_j_minus_1, C_i_j_plus_1, C_i_minus_1_j, C_i_plus_1_j, sum) schedule(static)
        for (int i = 1; i < x_chunk_size - 1; i++)
        {
            for (int j = 1; j < y_chunk_size - 1; j++)
            {
                x = -L * 0.5f + (i + x_start) * (L / N);
                y = -L * 0.5f + (j + y_start) * (L / N);
                u_at_pos = u * y;
                v_at_pos = v * x;
                C_i_j = C_n[i][j];

                C_i_minus_1_j = C_n[i - 1][j];
                C_i_j_minus_1 = C_n[i][j - 1];
                C_i_plus_1_j = C_n[i + 1][j];
                C_i_j_plus_1 = C_n[i][j + 1];

                // LAX method
                sum = C_i_j_minus_1 + C_i_j_plus_1 + C_i_minus_1_j + C_i_plus_1_j;
                C_n_1[i][j] = 0.25f * sum - 0.5f * del_t / del_x * (u_at_pos * (C_i_plus_1_j - C_i_minus_1_j) + v_at_pos * (C_i_j_plus_1 - C_i_j_minus_1));

                if (k == mid)
                    C_n_half[i][j] = C_n_1[i][j];
            }
        }
        double **temp = C_n;
        C_n = C_n_1;
        C_n_1 = temp;
    }
    // all timesteps done
    // store end time
    t2 = omp_get_wtime();

    // free some allocated heap memory
    free(send_down);
    free(send_up);
    free(send_left);
    free(send_right);
    free(recv_down);
    free(recv_up);
    free(recv_left);
    free(recv_right);

    // print time taken from rank 0
    if (mype == 0)
    {
        printf("Total time taken = %lf seconds\n", t2 - t1);
    }

    // write the mid array and the final array to the output file
    send_recv_data_and_write(C_n_half, mype, N, x_chunk_size, y_chunk_size, nprocs, &stat, x_n_chunks, y_n_chunks, file_start_end);
    send_recv_data_and_write(C_n, mype, N, x_chunk_size, y_chunk_size, nprocs, &stat, x_n_chunks, y_n_chunks, file_start_end);

    // finish and free up memory
    MPI_Finalize();

    if (mype == 0)
        fclose(file_start_end);
    free(C_n[0]);
    free(C_n_half[0]);
    free(C_n_1[0]);
    free(C_n);
    free(C_n_half);
    free(C_n_1);
    return 0;
}
