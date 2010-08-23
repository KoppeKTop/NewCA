#ifndef RNDGEN_H
#define RNDGEN_H

#define MM 0xFFFFFFFF
#define AA 1664525
#define CC 1013904223UL

#define MASK 0x3fffffff
#define MY_RAND_MAX MM
#include "genca.h"

__device__ uint4 * states=0;

__device__ inline unsigned int lcg_rand(unsigned int& seed)
{
    seed = seed * AA + CC;
    return seed;
}

__device__ inline unsigned int taus_rand_step(unsigned int& state, int S1, int S2, int S3, unsigned int M)
{
    unsigned int b = (((state << S1) ^ state) >> S2);
    state = (((state & M) << S3) ^ b);
    return state;
}

__device__ 
inline unsigned int hybrid_taus (uint4 & state)
{
    return 
        taus_rand_step(state.x, 13, 19, 12, 4294967294UL) ^
        taus_rand_step(state.y, 2, 25, 4, 4294967288UL) ^
        taus_rand_step(state.z, 3, 11, 17, 4294967280UL) ^
        lcg_rand(state.w);
}

__device__ RandomType randGPU(unsigned int idx)
{
	// Use classic random number generator
	// returns [0, MY_RAND_MAX]
	// X[n] = (aX[n-1] + c) mod m (only first 30 bits)
	// RandomType res = (RandomType)((AA * (long)seed + CC) % MM);
	// return res;

	// Use hubrid_taus gen
	return hybrid_taus(states[idx]);
}

void InitRandomGPU(unsigned int seed, const size_t cnt)
{
	if (states == 0) return;
	srand(seed);
	uint4 * buffer = new uint4[cnt];
	for (size_t ind = 0; ind < cnt; ++ind) {
	    buffer[ind].x = rand();
	    buffer[ind].y = rand();
	    buffer[ind].z = rand();
	    buffer[ind].w = rand();
	}
	size_t rand_size = cnt*sizeof(uint4);
	cudaMalloc((void**)&states, rand_size);
	cudaMemcpy(states, buffer, rand_size, cudaMemcpyHostToDevice);
	delete [] buffer;
}

void CleanRandomGPU()
{
    cudaFree(states);
}

#endif

