#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Versione con divisione esplicita del lavoro
double maxsum(int N, int LD, double *A, int NT) {
    
    double max_sum=-1, local_max, sum;
    int start, end, i, j;

    omp_set_num_threads(NT);

    #pragma omp parallel private(local_max, sum, start, end, i, j)
    {
        local_max = -1;
        start = omp_get_thread_num() * (N / NT);
        end = (omp_get_thread_num() + 1) * (N / NT);

        // Calcola il massimo locale
        for(i = start; i < end; ++i) {
            sum = 0.0;
            for(j = 0; j < N; ++j) {
                sum += sqrt(*(A + i*LD + j));
            }
            if(sum > local_max) {
                local_max = sum;
            }
        }
        
        // Aggiorna il massimo globale una sola volta per thread
        #pragma omp critical
        {
            if(local_max > max_sum) {
                max_sum = local_max;
            }
        }
    }
    return max_sum;
}

/*
// Versione con omp for
double maxsum(int N, int LD, double *A, int NT) {
    
    double max_sum=-1, local_max, sum;
    int i, j;

    omp_set_num_threads(NT);

    #pragma omp parallel private(local_max)
    {
        local_max = -1;

        // Calcola il massimo locale
        #pragma omp for private(j, sum) schedule(static) nowait
        for(i = 0; i < N; ++i) { // i è già private grazie a omp for
            sum = 0.0;
            for(j = 0; j < N; ++j) {
                sum += sqrt(*(A + i*LD + j));
            }
            if(sum > local_max) {
                local_max = sum;
            }
        }
        // Grazie a nowait qui non si crea una barriera implicita
        
        // Aggiorna il massimo globale una sola volta per thread
        #pragma omp critical
        {
            if(local_max > max_sum) {
                max_sum = local_max;
            }
        }
    }
    return max_sum;
}
*/