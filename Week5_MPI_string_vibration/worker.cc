#include <mpi.h>
#include "L.h"
// Finite difference method for stings
// d_(x, t+1) = L(x)*(d_(x+dx, t) + d_(x-dx, t))
//              + 2.0f*(1.0f-L(x))*(d_(x,t))
//              - d_(x, t-1) 

float * simulate(const float alpha, const long n_segments, const int n_steps, float *d_buf1, float *d_buf2, const int rank, const int world_size, const long segments_per_process) {

  float* d_t  = d_buf1;  // buffer for d(*, t)
  float* d_t1 = d_buf2;  // buffer for d(*, t+1)

  const long start_segment = segments_per_process*((long)rank)   +1L;
  const long last_segment  = segments_per_process*((long)rank+1L)+1L;

  const float dx = 1.0f/(float)n_segments;
  const float phase = 0.5f;
  MPI_Status stat;
  for(int t = 0; t < n_steps; t++) {
#pragma omp parallel for simd
    for(long i = start_segment; i < last_segment; i++) {
      const float L_x = L(alpha,phase,i*dx);
      d_t1[i] = L_x*(d_t[i+1] + d_t[i-1])
                +2.0f*(1.0f-L_x)*(d_t[i]) 
                - d_t1[i]; // The algorithm calls for d(i, t-1) here, but that is currently contained in d_t1
    }
    float* temp = d_t1; d_t1 = d_t; d_t=temp; // swap buffers
    
    //synchronize and gather segments data from other MPI processes
    //MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &d_t[1], segments_per_process, MPI_FLOAT, MPI_COMM_WORLD);
		if (!rank)
		{
			MPI_Send(&d_t[last_segment-1], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(&d_t[last_segment], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &stat);
		}
		else if (rank == world_size - 1)
		{
			MPI_Send(&d_t[start_segment], 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
			MPI_Recv(&d_t[start_segment-1], 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &stat);
		}
		else if (rank && !(rank % 2))
		{
			MPI_Send(&d_t[last_segment-1], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
			MPI_Send(&d_t[start_segment], 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
			MPI_Recv(&d_t[last_segment], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&d_t[start_segment-1], 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &stat);
		}
		else
		{
			MPI_Recv(&d_t[start_segment-1], 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&d_t[last_segment], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&d_t[start_segment], 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
			MPI_Send(&d_t[last_segment-1], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
		}
	}

	MPI_Gather(&d_t[start_segment], segments_per_process, MPI_FLOAT, &d_t1[1], segments_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
  d_t = d_t1;
	return d_t;
}
