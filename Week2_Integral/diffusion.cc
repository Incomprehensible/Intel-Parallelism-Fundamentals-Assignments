#include <mkl.h>
#include "distribution.h"

// n_particles 131072
// steps 500

//vectorize this function based on instruction on the lab page
/*int old_diffusion(const int n_particles, const int n_steps, const float x_threshold, const float alpha, VSLStreamStatePtr rnStream)
{
	int n_escaped=0;
	
	for (int i = 0; i < n_particles; i++)
	{
		float x = 0.0f;
		for (int j = 0; j < n_steps; j++)
		{
			float rn;
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, 1, &rn, -1.0, 1.0);
			x += dist_func(alpha, rn);
		}
		if (x > x_threshold) n_escaped++;
	}
	return n_escaped;
} */

#pragma omp declare simd
int how_many(const float threshold, const int n_particles, const float *x)
{
	int c = 0;
	for (int i = 0; i < n_particles; ++i)
	{
		if (x[i] > threshold)
			++c;
	}
	return c;
}

__attribute__((vector, nothrow))
int vsRngUniform(const MKL_INT, VSLStreamStatePtr, const MKL_INT, float [],  const float,  const float);

int diffusion(const int n_particles, const int n_steps, const float x_threshold, const float alpha, VSLStreamStatePtr rnStream)
{
	int	n_escaped = 0;
	float	x[n_particles];
	
	int rn_seq = 0;
	for (int j = 0; j < n_steps; ++j)
	{
		float rn[n_particles];
		vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, n_particles, rn, -1.0, 1.0);
		for (int i = 0; i < n_particles; ++i)
		{
			x[i] += dist_func(alpha, rn[i]);
		}
	}
	return (how_many(x_threshold, n_particles, x));
}
