#ifndef __cuda_2d_fdaco_H__
#define __cuda_2d_fdaco_H__

typedef struct {
	int nx;			// grid number for x-coordinate
	int nz;         // grid number for y-coordinate
	float dx;       // step length for x-coordinate
	float dz;       // step length for y-coordinate 
	float *rd;      // density
	float *rv;      // P-wave velocity
} Model;

typedef struct {
	int ns;         
	float sx;       // source location
	float sz;       // source location
	float f0;       // peak frequency
	float dt;       // time sample
	float *src;     // source amplitude
	int iss;        // source type
} Source;

#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

#endif

