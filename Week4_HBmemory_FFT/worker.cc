#include <mkl.h>
#include <hbwmalloc.h>
#include <iostream>
#include <stdio.h>

//implement scratch buffer on HBM and compute FFTs, refer instructions on Lab page
void runFFTs( const size_t fft_size, const size_t num_fft, MKL_Complex8 *data, DFTI_DESCRIPTOR_HANDLE *fftHandle) {

	MKL_Complex8 *hbw_data;
	hbw_posix_memalign((void **)&hbw_data, 4096, (sizeof(MKL_Complex8) * 1<<27));
	
	for(size_t i = 0; i < num_fft; i++)
	{
		#pragma omp parallel for
		for (size_t j = 0; j < fft_size; ++j) {
			hbw_data[j].real = data[i * fft_size + j].real;
			hbw_data[j].imag = data[i * fft_size + j].imag;
		}
		DftiComputeForward (*fftHandle, hbw_data);
		#pragma omp parallel for
		for (size_t j = 0; j < fft_size; ++j) {
			data[i * fft_size + j].real = hbw_data[j].real;
			data[i * fft_size + j].imag = hbw_data[j].imag;
		}
	}
}
