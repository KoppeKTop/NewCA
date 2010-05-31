#ifndef NEWCA_H
#define NEWCA_H

#include "genca.h"
#include <stddef.h>

typedef struct
{
	double D_A;
	double D_D;
	double D_E;
	double A_A;
	double A_E;
	double E_E;
	int neigh_cnt; 
    int max_iter;
    int n;
    int m;
    int drug_cnt;
    char * struct_filename;
    char * neg_filename;
    char * print_to;
    int count_every;
    FillType filling;
    int device;
    int save_bmp;
	
	int load_dmp;
	char * dmp_file;
} t_params;

extern "C" size_t get_sys_mem_sz(const t_params * params);
extern "C" size_t get_sys_sz(const t_params * params);

#endif
