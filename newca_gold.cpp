#include "newca.h"
#include <math.h>

size_t get_sys_mem_sz(const t_params * params)
{
	size_t res = get_sys_sz(params);
	res *= sizeof(ElementType);
	
	return res;
}

size_t get_sys_sz(const t_params * params)
{
	size_t res = (size_t)pow(params->n, DIMS);
	return res;
}

