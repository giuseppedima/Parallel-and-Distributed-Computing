#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>

void matmatikj(int ldA, int ldB, int ldC,
            double *A, double *B, double *C,
            int N1, int N2, int N3) {
    int i, k, j;
    for (i = 0; i < N1; ++i) {
        for (k = 0; k < N2; ++k) {
            for (j = 0; j < N3; ++j) {
                C[(i * ldC) + j] = C[(i * ldC) + j] + A[(i * ldA) + k] * B[(k * ldB) + j];
            }
        }
    }
}

void matmatblock(int ldA, int ldB, int ldC,
            double *A, double *B, double *C,
            int N1, int N2, int N3,
            int dbA, int dbB, int dbC) {
    
    int ii, jj, kk,
        Nii, Nkk, Njj,
        dbA_cur, dbB_cur, dbC_cur;

    // ceiling division (per avere un blocco in più se non divisibile)
    Nii = (N1 + dbA - 1) / dbA;
    Nkk = (N2 + dbB - 1) / dbB;
    Njj = (N3 + dbC - 1) / dbC;
    
    for (ii = 0; ii < Nii; ++ii) {
        dbA_cur = (ii != Nii-1) ? (dbA) : (N1 - (ii * dbA)); // l'ultimo blocco potrebbe essere più piccolo

        for (jj = 0; jj < Njj; ++jj) {
            dbC_cur = (jj != Njj-1) ? (dbC) : (N3 - (jj * dbC));

            for (kk = 0; kk < Nkk; ++kk) {
                dbB_cur = (kk != Nkk-1) ? (dbB) : (N2 - (kk * dbB));

                matmatikj(ldA, ldB, ldC,
                    A + (((ii * dbA) * ldA) + (kk * dbB)),
                    B + (((kk * dbB) * ldB) + (jj * dbC)),
                    C + (((ii * dbA) * ldC) + (jj * dbC)),
                    dbA_cur, dbB_cur, dbC_cur
                );
            }
        }
    }
}

void matmatthread(int ldA, int ldB, int ldC, 
    double *A, double *B, double *C,
    int N1, int N2, int N3,
    int dbA, int dbB, int dbC,
    int NTROW, int NTCOL) {
    
    int thread_num,
        row, col,
        rows_per_thread, cols_per_thread,
        row_start, col_start;

    #pragma omp parallel num_threads(NTROW*NTCOL) private(thread_num, row, col, rows_per_thread, cols_per_thread, row_start, col_start)
    {
        thread_num = omp_get_thread_num();
        
        row = thread_num / NTCOL;
        col = thread_num % NTCOL;

        rows_per_thread = N1 / NTROW;
        cols_per_thread = N3 / NTCOL;

        row_start = row * rows_per_thread;
        col_start = col * cols_per_thread;

        matmatblock(ldA, ldB, ldC,
            A + ((row_start * ldA) + 0),         // A(row_start, 0)
            B + ((0 * ldB) + col_start),         // B(0, col_start)
            C + ((row_start * ldC) + col_start), // C(row_start, col_start)
            (row != NTROW-1) ? (rows_per_thread) : (N1 - row_start), // l'ultima riga di thread prende anche le righe rimanenti
            N2, 
            (col != NTCOL-1) ? (cols_per_thread) : (N3 - col_start), // l'ultima colonna di thread prende anche le colonne rimanenti
            dbA, dbB, dbC
        );
    }
}

static int gcd(int a, int b) {
  int r;

  while (b) {
    r = a % b;
    a = b;
    b = r;
  }

  return a;
}

static int lcm(int a, int b) {
    if (a == 0 || b == 0) {
        return 0;
    }

    return (a / gcd(a, b)) * b;
}


void matmatdist(MPI_Comm Gridcom,
    int ldA, int ldB, int ldC,
    double *A, double *B, double *C,
    int N1, int N2, int N3,
    int DB1, int DB2, int DB3,
    int NTROW, int NTCOL) {
    
    int dims[2], periods[2], coords[2],
        directions[2],
        NProw, NPcol, K,
        N1loc, N2loc, N3loc,
        A_block_dim, B_block_dim,
        i, r, c, row;

    MPI_Comm Rowcom, Colcom;

    double *A_block, *B_block;

    MPI_Cart_get(Gridcom, 2, dims, periods, coords);
    NProw = dims[0];
    NPcol = dims[1];
    K = lcm(NProw, NPcol);

    N1loc = N1 / NProw;
    N2loc = N2 / K;
    N3loc = N3 / NPcol;

    directions[0] = 0;
    directions[1] = 1;
    MPI_Cart_sub(Gridcom, directions, &Rowcom);

    directions[0] = 1;
    directions[1] = 0;
    MPI_Cart_sub(Gridcom, directions, &Colcom);

    A_block_dim = N1loc * N2loc;
    B_block_dim = N2loc * N3loc;

    A_block = (double *)malloc(sizeof(double) * A_block_dim);
    B_block = (double *)malloc(sizeof(double) * B_block_dim);

    for (i = 0; i < K; ++i) {
        r = i % NProw;
        c = i % NPcol;

        if (coords[0] == r) {
            for (row = 0; row < N2loc; ++row) {
                memcpy(B_block + (row * N3loc),
                    B + (((i / NProw) * N2loc + row) * ldB),
                    sizeof(double) * N3loc
                );
            }
        }

        if (coords[1] == c) {
            for (row = 0; row < N1loc; ++row) {
                memcpy(A_block + (row * N2loc),
                    A + (row * ldA) + ((i / NPcol) * N2loc),
                    sizeof(double) * N2loc
                );
            }
        }

        MPI_Bcast(B_block, B_block_dim, MPI_DOUBLE, r, Colcom);
        MPI_Bcast(A_block, A_block_dim, MPI_DOUBLE, c, Rowcom);

        matmatthread(N2loc, N3loc, ldC,
            A_block, B_block, C,
            N1loc, N2loc, N3loc,
            DB1, DB2, DB3,
            NTROW, NTCOL
        );
    }

    free(A_block);
    free(B_block);

    MPI_Comm_free(&Rowcom);
    MPI_Comm_free(&Colcom);

}