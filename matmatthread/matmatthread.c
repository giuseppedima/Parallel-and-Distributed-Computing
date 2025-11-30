#include <omp.h>

void matmatijk(int ldA, int ldB, int ldC,
            double *A, double *B, double *C,
            int N1, int N2, int N3) {
    int i, j, k;
    for (i = 0; i < N1; ++i) {
        for (j = 0; j < N3; ++j) {
            for (k = 0; k < N2; ++k) {
                C[(i * ldC) + j] = C[(i * ldC) + j] + A[(i * ldA) + k] * B[(k * ldB) + j];
            }
        }
    }
}

void matmatjik(int ldA, int ldB, int ldC,
            double *A, double *B, double *C,
            int N1, int N2, int N3) {
    int j, i, k;
    for (j = 0; j < N3; ++j) {
        for (i = 0; i < N1; ++i) {
            for (k = 0; k < N2; ++k) {
                C[(i * ldC) + j] = C[(i * ldC) + j] + A[(i * ldA) + k] * B[(k * ldB) + j];
            }
        }
    }
}

void matmatikj(int ldA, int ldB, int ldC,
            double *A, double *B, double *C,
            int N1, int N2, int N3) {
    int i, k, j;
    for(i = 0; i < N1; ++i) {
        for(k = 0; k < N2; ++k) {
            for(j = 0; j < N3; ++j) {
                C[(i * ldC) + j] = C[(i * ldC) + j] + A[(i * ldA) + k] * B[(k * ldB) + j];
            }
        }
    }
}

void matmatjki(int ldA, int ldB, int ldC,
            double *A, double *B, double *C,
            int N1, int N2, int N3) {
    int i, j, k;
    for (j = 0; j < N3; ++j) {
        for (k = 0; k < N2; ++k) {
            for (i = 0; i < N1; ++i) {
                C[(i * ldC) + j] = C[(i * ldC) + j] + A[(i * ldA) + k] * B[(k * ldB) + j];
            }
        }
    }
}

void matmatkij(int ldA, int ldB, int ldC,
            double *A, double *B, double *C,
            int N1, int N2, int N3) {
    int k, i, j;
    for (k = 0; k < N2; ++k) {
        for (i = 0; i < N1; ++i) {
            for (j = 0; j < N3; ++j) {
                C[(i * ldC) + j] = C[(i * ldC) + j] + A[(i * ldA) + k] * B[(k * ldB) + j];
            }
        }
    }
}

void matmatkji(int ldA, int ldB, int ldC,
            double *A, double *B, double *C,
            int N1, int N2, int N3) {
    int k, j, i;
    for (k = 0; k < N2; ++k) {
        for (j = 0; j < N3; ++j) {
            for (i = 0; i < N1; ++i) {
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
