#ifndef RNDGEN_H
#define RNDGEN_H

#define MM 0xFFFFFFFF
#define AA 1664525
#define CC 1013904223

#define MASK 0x3fffffff
#define MY_RAND_MAX MM
#include "genca.h"

__device__ RandomType randGPU(RandomType seed)
{
	// Use classic random number generator
	// returns [0, MY_RAND_MAX]
	// X[n] = (aX[n-1] + c) mod m (only first 30 bits)
	RandomType res = (RandomType)((AA * (long)seed + CC) % MM);
	return res;
}

#endif

