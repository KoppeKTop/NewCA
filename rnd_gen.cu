#ifndef RNDGEN_H
#define RNDGEN_H

#define MM 0xFFFFFFFF
#define AA 1664525
#define CC 1013904223

#define MY_RAND_MAX MM
#include "genca.h"

__device__ RandomType randGPU(RandomType seed)
{
	// Use classic random number generator
	// X[n] = (aX[n-1] + c) mod m
	RandomType res = (AA * seed + CC) % MM;
	return res;
}

#endif

