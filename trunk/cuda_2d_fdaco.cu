/********************************************************************
*  
*  cuda_2d_fdaco.cu
*
*  This is an example of the CUDA program to calculate 2d acoustic 
*  wavefield using staggered-grid finite-difference like method with
*  PML absorbing boundary condition.
*
*  Scripted by:      Long Guihua
*  Initiated time:   2010/04/08
*  Last modified:    2010/04/08
*  E-mail:           longgh04@gmail.com
*  Address:          Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, 518055
*
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_2d_fdaco.h"

#define BLOCK_DIMX 16            // tile (and threadblock) size in x
#define BLOCK_DIMY 16            // tile (and threadblock) size in y
#define radius 4                 // length of difference coefficients
#define PI 3.1415926            

__constant__ float c_coeff[radius];

__global__ void fwd_2dhb_stg_orderN(float *g_input, float *g_output, float *g_param, const int dimx, 
							  const int dimy)
{
	__shared__ float s_data[BLOCK_DIMY + 2 * radius][BLOCK_DIMX + 2 * radius];

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int tx = threadIdx.x + radius;  // thread's x-index into corresponding shared memory tile 
	int ty = threadIdx.y + radius;  // thread's y-index into corresponding shared memory tile

	int in_idx = iy * dimx + ix;  // index for reading input
	
	// update the data in shared memory in halo part
	if (ix > radius - 1 && ix < dimx - radius && threadIdx.y < radius)  // halo above/below
	{
		s_data[threadIdx.y][tx] = g_input[in_idx - radius * dimx];
		s_data[threadIdx.y + BLOCK_DIMY + radius][tx] = g_input[in_idx + BLOCK_DIMY * dimx];
	}
//	if (iy > radius -1 && iy < dimy - radius && threadIdx.x < radius)  // halo left/right
	if (iy < dimy && threadIdx.x < radius)  // halo left/right
	{
		s_data[ty][threadIdx.x] = g_input[in_idx - radius];
		s_data[ty][threadIdx.x + BLOCK_DIMX + radius] = g_input[in_idx + BLOCK_DIMX];
	}
	
	// update the data in shared memory within BLOCKED part
	s_data[ty][tx] = g_input[in_idx];
	__syncthreads();

	// compute the output value
	float temp = 0.0f;
	for (int ic = 0; ic < radius; ic++)
		temp += c_coeff[ic]* (s_data[ty][tx + ic] - s_data[ty][tx - ic -1]);

	g_output[in_idx] = temp * g_param[in_idx];
}

__global__ void fwd_2dhf_stg_orderN(float *g_input, float *g_output, float *g_param, const int dimx, 
							  const int dimy)
{
	__shared__ float s_data[BLOCK_DIMY + 2 * radius][BLOCK_DIMX + 2 * radius];

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int tx = threadIdx.x + radius;  // thread's x-index into corresponding shared memory tile 
	int ty = threadIdx.y + radius;  // thread's y-index into corresponding shared memory tile

	int in_idx = iy * dimx + ix;  // index for reading input
	
	// update the data in shared memory in halo part
	if (ix > radius - 1 && ix < dimx - radius && threadIdx.y < radius)  // halo above/below
	{
		s_data[threadIdx.y][tx] = g_input[in_idx - radius * dimx];
		s_data[threadIdx.y + BLOCK_DIMY + radius][tx] = g_input[in_idx + BLOCK_DIMY * dimx];
	}
//	if (iy > radius -1 && iy < dimy - radius && threadIdx.x < radius)  // halo left/right
	if (iy < dimy && threadIdx.x < radius)  // halo left/right
	{
		s_data[ty][threadIdx.x] = g_input[in_idx - radius];
		s_data[ty][threadIdx.x + BLOCK_DIMX + radius] = g_input[in_idx + BLOCK_DIMX];
	}
	
	// update the data in shared memory within BLOCKED part
	s_data[ty][tx] = g_input[in_idx];
	__syncthreads();

	// compute the output value
	float temp = 0.0f;
	for (int ic = 0; ic < radius; ic++)
		temp += c_coeff[ic] * (s_data[ty][tx + ic + 1] - s_data[ty][tx - ic]);

	g_output[in_idx] = temp * g_param[in_idx];
}


__global__ void fwd_2dvb_stg_orderN(float *g_input, float *g_output, float *g_param, const int dimx, 
							  const int dimy)
{
	__shared__ float s_data[BLOCK_DIMY + 2 * radius][BLOCK_DIMX + 2 * radius];

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int tx = threadIdx.x + radius;  // thread's x-index into corresponding shared memory tile 
	int ty = threadIdx.y + radius;  // thread's y-index into corresponding shared memory tile

	int in_idx = iy * dimx + ix;  // index for reading input
	
	// update the data in shared memory in halo part
//	if (ix > radius - 1 && ix < dimx - radius && threadIdx.y < radius)  // halo above/below
	if (ix < dimx && threadIdx.y < radius)  // halo above/below
	{
		s_data[threadIdx.y][tx] = g_input[in_idx - radius * dimx];
		s_data[threadIdx.y + BLOCK_DIMY + radius][tx] = g_input[in_idx + BLOCK_DIMY * dimx];
	}
	if (iy > radius -1 && iy < dimy -radius && threadIdx.x < radius)  // halo left/right
	{
		s_data[ty][threadIdx.x] = g_input[in_idx - radius];
		s_data[ty][threadIdx.x + BLOCK_DIMX + radius] = g_input[in_idx + BLOCK_DIMX];
	}
	
	// update the data in shared memory within BLOCKED part
	s_data[ty][tx] = g_input[in_idx];
	__syncthreads();

	// compute the output value
	float temp = 0.0f;
	for (int ic = 0; ic < radius; ic++)
		temp += c_coeff[ic] * (s_data[ty + ic][tx] - s_data[ty - ic - 1][tx]);

	g_output[in_idx] = temp * g_param[in_idx];
}

__global__ void fwd_2dvf_stg_orderN(float *g_input, float *g_output, float *g_param, const int dimx, 
							  const int dimy)
{
	__shared__ float s_data[BLOCK_DIMY + 2 * radius][BLOCK_DIMX + 2 * radius];

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int tx = threadIdx.x + radius;  // thread's x-index into corresponding shared memory tile 
	int ty = threadIdx.y + radius;  // thread's y-index into corresponding shared memory tile

	int in_idx = iy * dimx + ix;  // index for reading input
	
	// update the data in shared memory in halo part
//	if (ix > radius - 1 && ix < dimx - radius && threadIdx.y < radius)  // halo above/below
	if (ix < dimx && threadIdx.y < radius)  // halo above/below
	{
		s_data[threadIdx.y][tx] = g_input[in_idx - radius * dimx];
		s_data[threadIdx.y + BLOCK_DIMY + radius][tx] = g_input[in_idx + BLOCK_DIMY * dimx];
	}
	if (iy > radius -1 && iy < dimy -radius && threadIdx.x < radius)  // halo left/right
	{
		s_data[ty][threadIdx.x] = g_input[in_idx - radius];
		s_data[ty][threadIdx.x + BLOCK_DIMX + radius] = g_input[in_idx + BLOCK_DIMX];
	}
	
	// update the data in shared memory within BLOCKED part
	s_data[ty][tx] = g_input[in_idx];
	__syncthreads();

	// compute the output value
	float temp = 0.0f;
	for (int ic = 0; ic < radius; ic++)
			temp += c_coeff[ic]* (s_data[ty + ic + 1][tx] - s_data[ty - ic][tx]);

	g_output[in_idx] = temp * g_param[in_idx];
}

__global__ void bd_fwd_2dhb_stg_orderN(float *g_input, float *g_output, float *g_param, const int dimx, 
							  const int dimy)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int in_idx = iy * dimx + ix;  // index for reading input

	if (ix == 0)   // set value 0.0f to the first column
		g_output[in_idx] = 0.0f;
	if (ix < radius && ix > 0 && iy < dimy ) // left boundary and backward
		g_output[in_idx] = (g_input[in_idx] - g_input[in_idx - 1]) * g_param[in_idx];
	if (ix > dimx - radius - 1 && ix < dimx && iy < dimy ) // right boundary and backward
		g_output[in_idx] = (g_input[in_idx] - g_input[in_idx - 1]) * g_param[in_idx];
}

__global__ void bd_fwd_2dhf_stg_orderN(float *g_input, float *g_output, float *g_param, const int dimx, 
							  const int dimy)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int in_idx = iy * dimx + ix;  // index for reading input

	if (ix < radius && iy < dimy ) // left boundary and forward
		g_output[in_idx] = (g_input[in_idx + 1] - g_input[in_idx]) * g_param[in_idx];
	if (ix > dimx - radius -1 && ix < dimx - 1 && iy < dimy ) // right boundary and forward
		g_output[in_idx] = (g_input[in_idx + 1] - g_input[in_idx]) * g_param[in_idx];
	if (ix == dimx - 1)
		g_output[in_idx] = 0.0f;
}

__global__ void bd_fwd_2dvb_stg_orderN(float *g_input, float *g_output, float *g_param, const int dimx, 
							  const int dimy)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int in_idx = iy * dimx + ix;  // index for reading input

	if (iy == 0)   // set value 0.0f to the first column
		g_output[in_idx] = 0.0f;
	if (iy < radius && iy > 0 && ix < dimx ) // left boundary and backward
		g_output[in_idx] = (g_input[in_idx] - g_input[in_idx - dimx]) * g_param[in_idx];
	if (iy > dimy - radius - 1 && iy < dimy && ix < dimx ) // right boundary and backward
		g_output[in_idx] = (g_input[in_idx] - g_input[in_idx - dimx]) * g_param[in_idx];
}

__global__ void bd_fwd_2dvf_stg_orderN(float *g_input, float *g_output, float *g_param, const int dimx, 
							  const int dimy)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int in_idx = iy * dimx + ix;  // index for reading input

	if (iy < radius && ix < dimx ) // left boundary and forward
		g_output[in_idx] = (g_input[in_idx + dimx] - g_input[in_idx]) * g_param[in_idx];
	if (iy > dimy - radius -1 && iy < dimy - 1 && ix < dimx ) // right boundary and forward
		g_output[in_idx] = (g_input[in_idx + dimx] - g_input[in_idx]) * g_param[in_idx];
	if (iy == dimy - 1)
		g_output[in_idx] = 0.0f;
}

__global__ void AddSource(int nxe, int nze, float *d_taux, float *d_tauz, float *d_src, float *d_tau)
{
	__shared__ float s_dtaux[BLOCK_DIMY][BLOCK_DIMX];
	__shared__ float s_dtauz[BLOCK_DIMY][BLOCK_DIMX];
	__shared__ float s_dsrc[BLOCK_DIMY][BLOCK_DIMX];

	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int idx = iy * nxe + ix;
	if (ix < nxe && iy < nze)
	{
		s_dtaux[threadIdx.y][threadIdx.x] = d_taux[idx];
		s_dtauz[threadIdx.y][threadIdx.x] = d_tauz[idx];
		s_dsrc[threadIdx.y][threadIdx.x] = d_src[idx];
	}
	d_tau[idx] = s_dtaux[threadIdx.y][threadIdx.x] + s_dtauz[threadIdx.y][threadIdx.x] + s_dsrc[threadIdx.y][threadIdx.x];
}

__global__ void MatMulAdd_PerElem(int nxe, int nze, float *c, float *a, float *abar, float *b, float alpha)
{
	__shared__ float s_a[BLOCK_DIMY][BLOCK_DIMX];
	__shared__ float s_abar[BLOCK_DIMY][BLOCK_DIMX];
	__shared__ float s_b[BLOCK_DIMY][BLOCK_DIMX];
	__shared__ float s_c[BLOCK_DIMY][BLOCK_DIMX];

	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	int idx = iy * nxe + ix;
	if (ix < nxe && iy < nze)
	{
		s_a[threadIdx.y][threadIdx.x] = a[idx];
		s_abar[threadIdx.y][threadIdx.x] = abar[idx];
		s_b[threadIdx.y][threadIdx.x] = b[idx];
		s_c[threadIdx.y][threadIdx.x] = c[idx];
	}
	c[idx] = s_a[threadIdx.y][threadIdx.x] * s_c[threadIdx.y][threadIdx.x]
		+ s_abar[threadIdx.y][threadIdx.x] * s_b[threadIdx.y][threadIdx.x] * alpha;
}

void submodext(int nx, int nz, int *abc, float *rd, float *rv, float *rde, float *rve)
{
	int ix, iz, nxe, nze;

	nxe = nx+abc[0]+abc[2];
	nze = nz+abc[1]+abc[3];

	/* model kernel */
	for (iz=abc[1]; iz<nz+abc[1]; iz++)
		for (ix=abc[0]; ix<nx+abc[0]; ix++) {
		rde[iz*nxe+ix] = rd[(iz-abc[1])*nx+ix-abc[0]];
		rve[iz*nxe+ix] = rv[(iz-abc[1])*nx+ix-abc[0]];
	}

	/* left- and right-sides */
	for (iz=abc[1]; iz<nz+abc[1]; iz++) {
		for (ix=0; ix<abc[0]; ix++) {
			rde[iz*nxe+ix] = rd[(iz-abc[1])*nx];
			rve[iz*nxe+ix] = rv[(iz-abc[1])*nx];
		}
		for (ix=nx+abc[0]; ix<nxe; ix++) {
			rde[iz*nxe+ix] = rd[(iz-abc[1])*nx+nx-1];
			rve[iz*nxe+ix] = rv[(iz-abc[1])*nx+nx-1];
		}
	}

	/* upper- and lower- sides */
	for (ix=abc[0]; ix<nx+abc[0]; ix++) {
		for (iz=0; iz<abc[1]; iz++) {
			rde[iz*nxe+ix] = rd[ix-abc[0]];
			rve[iz*nxe+ix] = rv[ix-abc[0]];
		}
		for (iz=nz+abc[1]; iz<nze; iz++) {
			rde[iz*nxe+ix] = rd[(nz-1)*nx+ix-abc[0]];
			rve[iz*nxe+ix] = rv[(nz-1)*nx+ix-abc[0]];
		}
	}

	/* upper-left corner */
	for (iz=0; iz<abc[1]; iz++)
		for (ix=0; ix<abc[0]; ix++) {
			rde[iz*nxe+ix] = rd[0];
			rve[iz*nxe+ix] = rv[0];
	}

	/* upper-right corner */
	for (iz=0; iz<abc[1]; iz++)
		for (ix=nx+abc[0]; ix<nxe; ix++) {
			rde[iz*nxe+ix] = rd[nx-1];
			rve[iz*nxe+ix] = rv[nx-1];
	}

	/* lower-left corner */
	for (iz=nz+abc[1]; iz<nze; iz++)
		for (ix=0; ix<abc[0]; ix++) {
			rde[iz*nxe+ix] = rd[(nz-1)*nx];
			rve[iz*nxe+ix] = rv[(nz-1)*nx];
	}
					
	/* lower-right corner */
	for (iz=nz+abc[1]; iz<nze; iz++)
		for (ix=nx+abc[0]; ix<nxe; ix++) {
			rde[iz*nxe+ix] = rd[(nz-1)*nx+nx-1];
			rve[iz*nxe+ix] = rv[(nz-1)*nx+nx-1];
	}
	return;
}

void subpml(int nx, int nz, float dx, float dz, float R, int *nmpl, float *ve, float *qx1, float *qz1, float *qx2, float *qz2)
{
	int i, j, nxe, nze;
	float tmp, idx;

	nxe = nx+nmpl[0]+nmpl[2];
	nze = nz+nmpl[1]+nmpl[3];

	for (j=0; j<nz; j++)
		for (i=0; i<nx; i++) {
		qx1[(j+nmpl[1])*nxe+i+nmpl[0]] = 0.0;
		qx2[(j+nmpl[1])*nxe+i+nmpl[0]] = 0.0;
		qz1[(j+nmpl[1])*nxe+i+nmpl[0]] = 0.0;
		qz2[(j+nmpl[1])*nxe+i+nmpl[0]] = 0.0;
		}

		tmp = (float)nmpl[0]*dx;
		for (j=0; j<nze; j++) /* left boundary */
			for (i=0; i<nmpl[0]; i++) {
				idx = (float)(nmpl[0]-i);
				qx1[j*nxe+i] = 3.0f*ve[j*nxe+i]*logf(1.0f/R)*(idx*dx/tmp)*(idx*dx/tmp)/(2.0f*tmp);
				qx2[j*nxe+i] = 3.0f*ve[j*nxe+i]*logf(1.0f/R)*((idx+0.5f)*dx/tmp)*((idx+0.5f)*dx/tmp)/(2.0f*tmp);
			}
		tmp = (float)nmpl[2]*dx;
		for (j=0; j<nze; j++) /* right boundary */
			for (i=nx+nmpl[0]; i<nxe; i++) {
				idx = (float)(i+nmpl[2]+1-nxe);
				qx1[j*nxe+i] = 3.0f*ve[j*nxe+i]*logf(1.0f/R)*(idx*dx/tmp)*(idx*dx/tmp)/(2.0f*tmp);
				qx2[j*nxe+i] = 3.0f*ve[j*nxe+i]*logf(1.0f/R)*((idx+0.5f)*dx/tmp)*((idx+0.5f)*dx/tmp)/(2.0f*tmp);
			}
		tmp = (float)nmpl[1]*dz;
		for (i=0; i<nxe; i++) /* upper boundary */
			for (j=0; j<nmpl[1]; j++) {
				idx = (float)(nmpl[1]-j);
				qz1[j*nxe+i] = 3.0f*ve[j*nxe+i]*logf(1.0f/R)*(idx*dz/tmp)*(idx*dz/tmp)/(2.0f*tmp);
				qz2[j*nxe+i] = 3.0f*ve[j*nxe+i]*logf(1.0f/R)*((idx+0.5f)*dz/tmp)*((idx+0.5f)*dz/tmp)/(2.0f*tmp);
			}
		tmp = (float)nmpl[3]*dz;
		for (i=0; i<nxe; i++) /* lower boundary */
			for (j=nz+nmpl[1]; j<nze; j++) {
				idx = (float)(j+nmpl[3]+1-nze);
				qz1[j*nxe+i] = 3.0f*ve[j*nxe+i]*logf(1.0f/R)*(idx*dz/tmp)*(idx*dz/tmp)/(2.0f*tmp);
				qz2[j*nxe+i] = 3.0f*ve[j*nxe+i]*logf(1.0f/R)*((idx+0.5f)*dz/tmp)*((idx+0.5f)*dz/tmp)/(2.0f*tmp);
		}

	return;
}

void substager(int nxe, int nze, float *rde, float *rve, float *bux, float *buz, float *kappa)
{
	int i, j;

	for (j=1; j<nze-1; j++)
		for (i=1; i<nxe-1; i++) {
		bux[j*nxe+i] = 2.0f/(rde[j*nxe+i]+rde[j*nxe+i+1]);
		buz[j*nxe+i] = 2.0f/(rde[j*nxe+i]+rde[(j+1)*nxe+i]);
		kappa[j*nxe+i] = rde[j*nxe+i]*rve[j*nxe+i]*rve[j*nxe+i];
		}

		for (j=0; j<nze; j++) {
			bux[j*nxe] = 1.0f/rde[j*nxe];
			buz[j*nxe] = 1.0f/rde[j*nxe];
			bux[j*nxe+nxe-1] = 1.0f/rde[j*nxe+nxe-1];
			buz[j*nxe+nxe-1] = 1.0f/rde[j*nxe+nxe-1];
			kappa[j*nxe] = rde[j*nxe]*rve[j*nxe]*rve[j*nxe];
			kappa[j*nxe+nxe-1] = rde[j*nxe+nxe-1]*rve[j*nxe+nxe-1]*rve[j*nxe+nxe-1];
		}

		for (i=0; i<nxe; i++) {	
			bux[i] = 1.0f/rde[i];
			buz[i] = 1.0f/rde[i];
			bux[(nze-1)*nxe+i] = 1.0f/rde[(nze-1)*nxe+i];
			buz[(nze-1)*nxe+i] = 1.0f/rde[(nze-1)*nxe+i];
			kappa[i] = rde[i]*rve[i]*rve[i];
			kappa[(nze-1)*nxe+i] = rde[(nze-1)*nxe+i]*rve[(nze-1)*nxe+i]*rve[(nze-1)*nxe+i];
		}

	return;
}

void wavelet(Source sour)
{
	float t0 = 1.5f*sqrtf(6.0f)/((float)PI*sour.f0);
	
	float t, da, da2;
	for (int i=0; i<sour.ns; i++) 
	{
		t = (float)i*sour.dt;
		da = (float)PI*sour.f0*(t-t0);
		da2 = da*da;
		if (sour.iss == 1) sour.src[i] = cosf(2.0f*(float)PI*sour.f0*t);
		else if (sour.iss == 2) sour.src[i] = (1.0f-2.0f*da2)*expf(-da2);
		else if (sour.iss == 3) sour.src[i] = (t-t0)*expf(-da2);
		else sour.src[i] = -4.0f*da*(float)PI*sour.f0*expf(-da2)
				-2.0f*da*(float)PI*sour.f0*(1.0f-2.0f*da2)*expf(-da2);
	}
	return;
}

void forward(float t, Model model, Source sour, int *abc, float R, float tpoint, float *snap, float *seis)
{
	// Extend model grids
	int nxe = model.nx + abc[0] + abc[2];
	int nze = model.nz + abc[1] + abc[3];

	// Allocate memory for model parameters
	float *rde = (float *)malloc(nxe * nze * sizeof(float));
	float *rve = (float *)malloc(nxe * nze * sizeof(float));
	float *kappa = (float *)malloc(nxe * nze * sizeof(float));
	float *bux = (float *)malloc(nxe * nze * sizeof(float));
	float *buz = (float *)malloc(nxe * nze * sizeof(float));

	// Extend model grids
	submodext(model.nx, model.nz, abc, model.rd, model.rv, rde, rve);

	// Interpolate model parameters in-between grids
	substager(nxe, nze, rde, rve, bux, buz, kappa);

	// Load model parameters from host to device
	float *d_bux;
	cudaMalloc((void **) &d_bux, nxe * nze * sizeof(float));
	cudaMemcpy(d_bux, bux, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *d_buz;
	cudaMalloc((void **) &d_buz, nxe * nze * sizeof(float));
	cudaMemcpy(d_buz, buz, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *d_kappa;
	cudaMalloc((void **) &d_kappa, nxe * nze * sizeof(float));
	cudaMemcpy(d_kappa, kappa, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	// Free memory on host
	free(bux); free(buz); free(kappa);

	// Allocate memory for PML boundary matrices
	float *pmlx = (float *)malloc(nxe * nze * sizeof(float));
	float *pmlz = (float *)malloc(nxe * nze * sizeof(float));
	float *pmlxh = (float *)malloc(nxe * nze * sizeof(float));
	float *pmlzh = (float *)malloc(nxe * nze * sizeof(float));

	// Generate PML boundary governing matrices
	subpml(model.nx, model.nz, model.dx, model.dz, R, abc, rve, pmlx, pmlz, pmlxh, pmlzh);

	// Define and generate temper matrices of PML boundary 
	float *tempx = (float *)malloc(nxe * nze * sizeof(float));
	float *tempz = (float *)malloc(nxe * nze * sizeof(float));
	float *tempxh = (float *)malloc(nxe * nze * sizeof(float));
	float *tempzh = (float *)malloc(nxe * nze * sizeof(float));
	float *tempx5 = (float *)malloc(nxe * nze * sizeof(float));
	float *tempz5 = (float *)malloc(nxe * nze * sizeof(float));
	float *tempxh5 = (float *)malloc(nxe * nze * sizeof(float));
	float *tempzh5 = (float *)malloc(nxe * nze * sizeof(float));

	for (int iz = 0; iz < nze; iz++)
	{
		for (int ix = 0; ix < nxe; ix++)
		{
			tempx[iz * nxe + ix] = expf(-sour.dt * pmlx[iz * nxe + ix]);
			tempz[iz * nxe + ix] = expf(-sour.dt * pmlz[iz * nxe + ix]);
			tempxh[iz * nxe + ix] = expf(-sour.dt * pmlxh[iz * nxe + ix]);
			tempzh[iz * nxe + ix] = expf(-sour.dt * pmlzh[iz * nxe + ix]);
			tempx5[iz * nxe + ix] = expf(-0.5f * sour.dt * pmlx[iz * nxe + ix]);
			tempz5[iz * nxe + ix] = expf(-0.5f * sour.dt * pmlz[iz * nxe + ix]);
			tempxh5[iz * nxe + ix] = expf(-0.5f * sour.dt * pmlxh[iz * nxe + ix]);
			tempzh5[iz * nxe + ix] = expf(-0.5f * sour.dt * pmlzh[iz * nxe + ix]);
		}
	}

	// Load PML temper matrices from host to device
	float *d_tempx; 
	cudaMalloc((void **) &d_tempx, nxe * nze * sizeof(float));
	cudaMemcpy(d_tempx, tempx, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);
	float *d_tempz; 
	cudaMalloc((void **) &d_tempz, nxe * nze * sizeof(float));
	cudaMemcpy(d_tempz, tempz, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);
	float *d_tempxh; 
	cudaMalloc((void **) &d_tempxh, nxe * nze * sizeof(float));
	cudaMemcpy(d_tempxh, tempxh, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);
	float *d_tempzh; 
	cudaMalloc((void **) &d_tempzh, nxe * nze * sizeof(float));
	cudaMemcpy(d_tempzh, tempzh, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);
	float *d_tempx5; 
	cudaMalloc((void **) &d_tempx5, nxe * nze * sizeof(float));
	cudaMemcpy(d_tempx5, tempx5, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);
	float *d_tempz5; 
	cudaMalloc((void **) &d_tempz5, nxe * nze * sizeof(float));
	cudaMemcpy(d_tempz5, tempz5, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);
	float *d_tempxh5; 
	cudaMalloc((void **) &d_tempxh5, nxe * nze * sizeof(float));
	cudaMemcpy(d_tempxh5, tempxh5, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);
	float *d_tempzh5; 
	cudaMalloc((void **) &d_tempzh5, nxe * nze * sizeof(float));
	cudaMemcpy(d_tempzh5, tempzh5, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	// Free memory on host
	free(pmlx); free(pmlz); free(pmlxh); free(pmlzh);
	free(tempx); free(tempz); free(tempxh); free(tempzh);
	free(tempx5); free(tempz5); free(tempxh5); free(tempzh5);

	// Define zero vector with lenghth nxe x nze
	float *zero = (float *)malloc(nxe * nze * sizeof(float));
	for (int iz = 0; iz < nze; iz++)
		for (int ix = 0; ix < nxe; ix++)
			zero[iz * nxe + ix] = 0.0f;

	// Define stress and strain vector on device and initialization
	float *d_tau;
	cudaMalloc((void **) &d_tau, nxe * nze * sizeof(float));
	cudaMemcpy(d_tau, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *d_taux;
	cudaMalloc((void **) &d_taux, nxe * nze * sizeof(float));
	cudaMemcpy(d_taux, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *d_tauz;
	cudaMalloc((void **) &d_tauz, nxe * nze * sizeof(float));
	cudaMemcpy(d_tauz, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *d_velx;
	cudaMalloc((void **) &d_velx, nxe * nze * sizeof(float));
	cudaMemcpy(d_velx, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *d_velz;
	cudaMalloc((void **) &d_velz, nxe * nze * sizeof(float));
	cudaMemcpy(d_velz, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	// Define wavefield differentiation on device
	float *dtau_x;
	cudaMalloc((void **) &dtau_x, nxe * nze * sizeof(float));
	cudaMemcpy(dtau_x, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *dtau_z;
	cudaMalloc((void **) &dtau_z, nxe * nze * sizeof(float));
	cudaMemcpy(d_tau, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *dvel_x;
	cudaMalloc((void **) &dvel_x, nxe * nze * sizeof(float));
	cudaMemcpy(dvel_x, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	float *dvel_z;
	cudaMalloc((void **) &dvel_z, nxe * nze * sizeof(float));
	cudaMemcpy(dvel_z, zero, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

	// Define pressure wavefield on host
	float *tau = (float *)malloc(nxe * nze * sizeof(float));

	// Define source matrix on host and initiate
	float *h_src = (float *)malloc(nxe * nze * sizeof(float));
	for (int iz = 0; iz < nze; iz++)
		for (int ix = 0; ix < nxe; ix++)
			h_src[iz * nxe + ix] = 0.0f;

	// Define source matrix on device
	float *d_src;
	cudaMalloc((void **) &d_src, nxe * nze * sizeof(float));

	// Total time step for wavefield to propagate
	int tot = NINT(t / sour.dt);

	// Time point to store the wavefield snapshot
	int tp = NINT(t / sour.dt) - 1;

	// Source location
	int isx = NINT(sour.sx / model.dx) + abc[0];
	int isz = NINT(sour.sz / model.dz) + abc[1];

	// dtdx and dtdz
	float dtdx = sour.dt / model.dx;
	float dtdz = sour.dt / model.dz;

	// time loop for wavefield propagating
	for (int it = 0; it < tot; it++)
	{
		float sampl;
		if ( it < sour.ns) 
			sampl = sour.src[it];
		else sampl = 0.0f;

		// Build source matrix
		h_src[isz * nxe + isx] = sampl;

		cudaMemcpy(d_src, h_src, nxe * nze * sizeof(float), cudaMemcpyHostToDevice);

		dim3 dimBlock(BLOCK_DIMX, BLOCK_DIMY);
		dim3 dimGrid(nxe / BLOCK_DIMX, nze / BLOCK_DIMY);

        fwd_2dhf_stg_orderN<<<dimGrid, dimBlock>>>(d_velx, dvel_x, d_kappa, nxe, nze);
		bd_fwd_2dhf_stg_orderN<<<dimGrid, dimBlock>>>(d_velx, dvel_x, d_kappa, nxe, nze);
		fwd_2dvf_stg_orderN<<<dimGrid, dimBlock>>>(d_velz, dvel_z, d_kappa, nxe, nze);
		bd_fwd_2dvf_stg_orderN<<<dimGrid, dimBlock>>>(d_velz, dvel_z, d_kappa, nxe, nze);

		MatMulAdd_PerElem<<<dimGrid, dimBlock>>>(nxe, nze, d_taux, d_tempx, d_tempx5, dvel_x, dtdx);
		MatMulAdd_PerElem<<<dimGrid, dimBlock>>>(nxe, nze, d_tauz, d_tempz, d_tempz5, dvel_z, dtdz);
//		MatMulAdd_PerElem<<<dimGrid, dimBlock>>>(nxe, nze, d_taux, d_tempx, d_tempx5, dvel_x, dtdx);
//		MatMulAdd_PerElem<<<dimGrid, dimBlock>>>(nxe, nze, d_tauz, d_tempz, d_tempz5, dvel_z, dtdz);

		AddSource<<<dimGrid, dimBlock>>>(nxe, nze, d_taux, d_tauz, d_src, d_tau);

		fwd_2dhb_stg_orderN<<<dimGrid, dimBlock>>>(d_tau, dtau_x, d_bux, nxe, nze);
		bd_fwd_2dhb_stg_orderN<<<dimGrid, dimBlock>>>(d_tau, dtau_x, d_bux, nxe, nze);
		fwd_2dvb_stg_orderN<<<dimGrid, dimBlock>>>(d_tau, dtau_z, d_buz, nxe, nze);
		bd_fwd_2dvb_stg_orderN<<<dimGrid, dimBlock>>>(d_tau, dtau_z, d_buz, nxe, nze);

		MatMulAdd_PerElem<<<dimGrid, dimBlock>>>(nxe, nze, d_velx, d_tempx, d_tempx5, dtau_x, dtdx);
		MatMulAdd_PerElem<<<dimGrid, dimBlock>>>(nxe, nze, d_velz, d_tempz, d_tempz5, dtau_z, dtdz);
//		MatMulAdd_PerElem<<<dimGrid, dimBlock>>>(nxe, nze, d_velx, d_tempxh, d_tempxh5, dtau_x, dtdx);
//		MatMulAdd_PerElem<<<dimGrid, dimBlock>>>(nxe, nze, d_velz, d_tempzh, d_tempzh5, dtau_z, dtdz);

		// Save snapshot at time tp
		cudaMemcpy(tau, d_tau, nxe * nze * sizeof(float), cudaMemcpyDeviceToHost);
		for (int ix = abc[0]; ix < model.nx + abc[0]; ix++)
			seis[it * model.nx + ix - abc[0]] = tau[abc[1] * nxe + ix];
		if (it == tp)
		{
			for (int iz = abc[1]; iz < model.nz + abc[1]; iz++)
				for (int ix = abc[0]; ix < model.nx + abc[0]; ix++)
					snap[(iz-abc[1]) * model.nx + ix - abc[0]] = tau[iz * nxe + ix];
		}	
	}
	
	// Free memory on device
	cudaFree(dtau_x); cudaFree(dtau_z); 
	cudaFree(dvel_x); cudaFree(dvel_z);
	cudaFree(d_taux); cudaFree(d_tauz);
	cudaFree(d_velx); cudaFree(d_velz);
	cudaFree(d_tau); cudaFree(d_kappa);
	cudaFree(d_bux); cudaFree(d_buz);
	cudaFree(d_src);
	cudaFree(d_tempx); cudaFree(d_tempz);
	cudaFree(d_tempxh); cudaFree(d_tempzh);
	cudaFree(d_tempx5); cudaFree(d_tempz5);
	cudaFree(d_tempxh5); cudaFree(d_tempzh5);

	// Free memory on host
	free(rde); free(rve);
	free(tau); free(h_src);
}

#define row (256 - 40)
#define col (512 - 40)

int main(void)
{
	// Differentiation coefficients
	float h_coeff[radius] = {1225.f/1024.0f, -245.f/3072.f, 49.f/5120.f, -5.f/7168.f};
	
	cudaMemcpyToSymbol(c_coeff, h_coeff, radius * sizeof(float));
	if ( cudaGetLastError() != cudaSuccess )
	{
		printf("coefficient upload to GPU failed \n");
		exit(-3);
	}

	// set time and boundary
	float t = 3.0f;
	float tp = 1.0f;
	float R = 1.0e-3f;
	int abc[4] = {20, 20, 20, 20};

	// set model parameters
	Model model;
	model.nx = col;
	model.nz = row;
	model.dx = 10.0f;
	model.dz = 10.0f;
	
	model.rd = (float *)malloc(model.nx * model.nz * sizeof(float));
	model.rv = (float *)malloc(model.nx * model.nz * sizeof(float));
	int Lx = model.nx + abc[0] + abc[2];
	int Lz = model.nz + abc[1] + abc[3];

	for (int iz = 0; iz < model.nz; iz ++)
	{
		for (int ix = 0; ix < model.nx; ix++)
		{
			model.rd[iz * model.nx + ix] = 1500.0f;
			if (iz > 3 * Lz /8 - abc[1] && ix > 3 * Lx / 8 -abc[0])
				model.rv[iz * model.nx + ix] = 3500.0f;
			else
				model.rv[iz * model.nx + ix] = 1500.0f;
		}
	}

	// Set source parameters
	Source sour;
	sour.ns = 512;
//	sour.sx = (3 * Lx / 8 - abc[0]) * model.dx;
//	sour.sz = (Lz / 4 - abc[1]) * model.dz;
	sour.sx = (Lx * 1 / 2  - abc[0]) * model.dx;
	sour.sz = (Lz * 1 / 2 - abc[1]) * model.dz * 0.0f;
	sour.dt = 0.001;
	sour.f0 = 15.0;
	sour.iss = 2;
	sour.src = (float *)malloc(sour.ns * sizeof(float));
	wavelet(sour);

	// Allocate memory for snapshot wavefield
	float *snap = (float *)malloc(model.nx * model.nz * sizeof(float));
	float *seis = (float *)malloc(model.nx * NINT(t / sour.dt) * sizeof(float));

	clock_t start = clock();
	forward(t, model, sour, abc, R, tp, snap, seis);
	clock_t end = clock();

	printf("Time for computation of size %d x %d is %es\n", model.nx, model.nz, (double)(end - start) / CLOCKS_PER_SEC);

	FILE *fsnap = fopen("fsnap.dat", "w");
	for (int ix = 0; ix < model.nx; ix++)
		for (int iz = 0; iz < model.nz; iz++)
			fprintf(fsnap, "%e\n", snap[iz * model.nx + ix]);
	fclose(fsnap);

	printf("nstep = %d\n", NINT(t / sour.dt));
	FILE * fseis = fopen("fseis.dat","w");
	for (int ix = 0; ix < model.nx; ix ++)
		for (int it = 0; it < NINT(t / sour.dt); it ++)
			fprintf(fseis, "%e\n", seis[it * model.nx + ix]);
	fclose(fseis);


	// Free memory
	free(model.rd);
	free(model.rv);
	free(sour.src);
	free(snap); free(seis);
	
	return 0;
}
