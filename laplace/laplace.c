#include <mpi.h>

void laplace(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter){

    int rank, NP, rowCount, iter, i, j;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NP);

    rowCount = N/NP;

    for(iter=0; iter<Niter; ++iter) {

        if(rank!=0) {
            MPI_Send(A, N, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
            MPI_Recv(daprev, N, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, NULL);
        }

        if(rank!=NP-1) {
            MPI_Send(A+(rowCount-1)*LD, N, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD);
            MPI_Recv(danext, N, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, NULL);
        }

        for(i=1;i<=rowCount-2;++i) {
            for(j=1;j<=N-2;++j) {
                B[i*LD+j] = (A[(i-1)*LD+j] + A[(i+1)*LD+j] + A[i*LD+(j-1)] + A[i*LD+(j+1)]) * 0.25;
            }
        }

        if(rank!=0) {
            for(j=1;j<=N-2;++j) {
                B[0*LD+j] = (daprev[j] + A[1*LD+j] + A[0*LD+(j-1)] + A[0*LD+(j+1)]) * 0.25;
            }
        }

        if(rank!=NP-1) {
            for(j=1;j<=N-2;++j) {
                B[(rowCount-1)*LD+j] = (A[(rowCount-2)*LD+j] + danext[j] + A[(rowCount-1)*LD+(j-1)] + A[(rowCount-1)*LD+(j+1)]) * 0.25;
            }
        }

        for(i=1;i<=rowCount-2;++i) {
            for(j=1;j<=N-2;++j) {
                A[i*LD+j] = B[i*LD+j];
            }
        }

        if(rank!=0) {
            for(j=1;j<=N-2;++j) {
                A[0*LD+j] = B[0*LD+j];
            }
        }

        if(rank!=NP-1) {
            for(j=1;j<=N-2;++j) {
                A[(rowCount-1)*LD+j] = B[(rowCount-1)*LD+j];
            }
        }

    }
}

void laplace_nb(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter){

    int rank, NP, rowCount, iter, i, j;
    MPI_Request req_send_prev, req_send_next, req_recv_prev, req_recv_next;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NP);

    rowCount = N/NP;

    for(iter=0; iter<Niter; ++iter) {

        if(rank!=0) {
            MPI_Isend(A, N, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &req_send_prev);
            MPI_Irecv(daprev, N, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, &req_recv_prev);
        }

        if(rank!=NP-1) {
            MPI_Isend(A+(rowCount-1)*LD, N, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &req_send_next);
            MPI_Irecv(danext, N, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &req_recv_next);
        }

        for(i=1;i<=rowCount-2;++i) {
            for(j=1;j<=N-2;++j) {
                B[i*LD+j] = (A[(i-1)*LD+j] + A[(i+1)*LD+j] + A[i*LD+(j-1)] + A[i*LD+(j+1)]) * 0.25;
            }
        }

        if(rank!=0) {
            MPI_Wait(&req_recv_prev, NULL);
            for(j=1;j<=N-2;++j) {
                B[0*LD+j] = (daprev[j] + A[1*LD+j] + A[0*LD+(j-1)] + A[0*LD+(j+1)]) * 0.25;
            }
        }

        if(rank!=NP-1) {
            MPI_Wait(&req_recv_next, NULL);
            for(j=1;j<=N-2;++j) {
                B[(rowCount-1)*LD+j] = (A[(rowCount-2)*LD+j] + danext[j] + A[(rowCount-1)*LD+(j-1)] + A[(rowCount-1)*LD+(j+1)]) * 0.25;
            }
        }

        for(i=1;i<=rowCount-2;++i) {
            for(j=1;j<=N-2;++j) {
                A[i*LD+j] = B[i*LD+j];
            }
        }

        if(rank!=0) {
            MPI_Wait(&req_send_prev, NULL);
            for(j=1;j<=N-2;++j) {
                A[0*LD+j] = B[0*LD+j];
            }
        }

        if(rank!=NP-1) {
            MPI_Wait(&req_send_next, NULL);
            for(j=1;j<=N-2;++j) {
                A[(rowCount-1)*LD+j] = B[(rowCount-1)*LD+j];
            }
        }

    }
}