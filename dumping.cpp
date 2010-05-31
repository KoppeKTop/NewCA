/*
 *  dumping.c
 *  
 *
 *  Created by Andrey on 27/5/10.
 *  Copyright 2010. All rights reserved.
 *
 */

#include "dumping.h"
#include "genca.h"
#include <stdio.h>

extern "C" void save_dump(const t_params * params, const ElementType * src)
{
	char * filename = new char[256];
	sprintf(filename, "dump_%.1f_%.1f.dmp", params->D_A, params->D_D);
	FILE * out = fopen(filename, "w");
	if (out == NULL) {
		fprintf(stderr, "Can t open file to dump\n");
		return;
	}
	fwrite(src, sizeof(ElementType), get_sys_sz(params), out);
	fclose(out);
}

extern "C" int load_dump(const t_params * params, ElementType * dst)
{
	FILE * in = fopen(params->dmp_file, "r");
	if (in == 0) {
		fprintf(stderr, "Can t read from file\n");
		return 1;
	}
	fread(dst, sizeof(ElementType), get_sys_sz(params), in);
	fclose(in);
	return 0;
}
