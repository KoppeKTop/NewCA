/* 
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include "dumping.h"

#include <cuda.h>
#include "newca.h"
// includes, project
#include "cutil_inline.h"
// includes, kernels
#include "newca_kernel.cu"
#include "load_params.h"
#include "FieldSaver.h"


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runCA( int argc, char** argv);
int GetDrugCount(t_params * params, ElementType * fld);
dim3 GetDims(t_params * params);
void CountAround(t_params * params, ElementType * fld, const CoordVec * neight_map, int x, int y, int z, CoordVec * point_list);
void Log(t_params * params, char * str);
void ClearLog(t_params * params);
void PrintParams(t_params * params);

//extern "C"
//void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runCA( argc, argv);

    exit(EXIT_SUCCESS);
}

float frand()
{
	float res = (float)rand()/(float)RAND_MAX;
	#ifdef _DEBUG
	if (res < 0.0f || 1.0f <= res)
	{
		fprintf(stderr, "Wrong rand: %f\n", res);
	} 
	#endif
    return res;
}

void generate_rnd(RandomType * rnd, size_t sz)
{
	for (size_t i = 0; i < sz; ++i)
		rnd[i] = rand();
}

void cp_rnd_dev(const RandomType * rnd, RandomType * dev_rnd, size_t mem_sz)
{
	cutilSafeCall( cudaMemcpy( dev_rnd, rnd, mem_sz,
                                cudaMemcpyHostToDevice) );
}

void
runCA( int argc, char** argv) 
{
    srand(time(NULL));
    char * config_file = new char[256];
    
    if (argc != 2 || !file_exists(argv[1]))
    {
        printf("Usage: %s config_file\n", argv[0]);
        return;
    }
    strncpy(config_file, argv[1], 255);
    
    t_params * params = new t_params;
    int err = get_params(params, config_file);
    if (err != 0)
    {
        return;
    }
    
    ClearLog(params);
    PrintParams(params);
        
    char * str_buf = new char[512];
    str_buf[0] = '\0';
    
    printf("Device %d\n", params->device);
	char * dev_str = new char [255];
	sprintf(dev_str, "-device=%d", params->device);
	// delete [] argv[1];
	argv[1] = dev_str;
    CUT_DEVICE_INIT(argc, argv);
    cutilCheckMsg("CUDA init error");
    
    dim3 dim_len(params->n, params->n, params->n);
    
    // setup execution parameters
    dim3  threads( 16, 16, 1);
    dim3  grid( dim_len.x/(2*threads.x) + 1, dim_len.y/(2*threads.y) + 1, 1);
    
    size_t mem_size = dim_len.x*dim_len.y*sizeof(ElementType);
    size_t random_elements = dim_len.x*dim_len.y/4;
    size_t random_mem_size = random_elements*sizeof(RandomType);
    size_t weight_size = LABEL_LAST*LABEL_LAST*sizeof(float);
    size_t rot_size = dim_len.x*dim_len.y*sizeof(RotationType)/4;
    
    // allocate host memory
    RandomType * h_random = (RandomType*) malloc(random_mem_size);
    generate_rnd(h_random, random_elements);

    float * h_weights = (float*) malloc(weight_size);
    memset(h_weights, 0, weight_size);
    RotationType * h_rotability_even = (RotationType*) malloc(rot_size);
    RotationType * h_rotability_odd = (RotationType*) malloc(rot_size);
    Coord::SetDefDims(DIMS);
    CellsField * fld = new CellsField(Coord(dim_len.x, dim_len.y, dim_len.z), Coord());
    fld->Fill(0);
    
    // initalize the memory
	if (params->load_dmp)
	{
		err = load_dump(params, (ElementType *)fld->GetCells());
		if (err)
		{
			return;
		}
	}
	else
	{
		fill_with_structure(params, fld);
		fill_with_drug(params, fld);
		get_rotation_maps(fld, h_rotability_even, h_rotability_odd);
	}
    const ElementType* h_idata = fld->GetCells();
    
#ifdef M_DEBUG
    
    printf("h_rotability_even: \n");
    for (int j = 0; j < dim_len.y/2; ++j)
    {
        for (int i = 0; i < dim_len.x/2; ++i)
        {
            printf("%d \t", h_rotability_even[i + j*dim_len.x/2]);
        }
        printf("\n");
    }
    
    printf("h_rotability_odd: \n");
    for (int j = 0; j < dim_len.y/2; ++j)
    {
        for (int i = 0; i < dim_len.x/2; ++i)
        {
            printf("%d \t", h_rotability_odd[i + j*dim_len.x/2]);
        }
        printf("\n");
    }
#endif
    
    for (int i = 0; i < LABEL_LAST*LABEL_LAST; ++i)
    {
        h_weights[i] = 0;
    }
    h_weights[LABEL_DRUG + LABEL_LAST*LABEL_DRUG] = (float)params->D_D;
    h_weights[LABEL_DRUG + LABEL_LAST*LABEL_AG] = (float)params->D_A;
    h_weights[LABEL_AG + LABEL_LAST*LABEL_DRUG] = (float)params->D_A;
    h_weights[LABEL_EMPTY + LABEL_LAST*LABEL_DRUG] = (float)params->D_E;
    h_weights[LABEL_DRUG + LABEL_LAST*LABEL_EMPTY] = (float)params->D_E;
    h_weights[LABEL_EMPTY + LABEL_LAST*LABEL_EMPTY] = (float)params->E_E;
    h_weights[LABEL_EMPTY + LABEL_LAST*LABEL_AG] = (float)params->A_E;
    h_weights[LABEL_AG + LABEL_LAST*LABEL_EMPTY] = (float)params->A_E;
    //cutilSafeCall(cudaMemcpyToSymbol(weights, h_weights, weight_size));
    float * weights = NULL;
    cutilSafeCall(cudaMalloc((void**) &weights, weight_size));
    cutilSafeCall(cudaMemcpy(weights, h_weights, weight_size, 
                             cudaMemcpyHostToDevice));
    // allocate device memory
    ElementType * d_idata = 0;
    cutilSafeCall(cudaMalloc((void **) &d_idata, mem_size));
    
    RandomType * d_random = 0;
    cutilSafeCall( cudaMalloc( (void**) &d_random, random_mem_size));
    cp_rnd_dev(h_random, d_random, random_mem_size);
    
    RotationType * d_rotability_even = 0;
    cutilSafeCall( cudaMalloc( (void**) &d_rotability_even, rot_size));
    RotationType * d_rotability_odd = 0;
    cutilSafeCall( cudaMalloc( (void**) &d_rotability_odd, rot_size));
    cutilCheckMsg("Malloc error");
    
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );
    //cutilSafeCall(cudaMemcpy2D(d_idata, pitched_sz, h_idata, standart_pitch, 
    //             standart_pitch, dim_len.y, cudaMemcpyHostToDevice));
    
    ElementType * h_odata = (ElementType*) malloc(mem_size);
    
    memset(h_odata, 0xAA, mem_size);
    
    ElementType* d_odata = 0;
    //cutilSafeCall( cudaMallocPitch( (void**) &d_odata, &pitched_sz, standart_pitch, dim_len.y));
    cutilSafeCall( cudaMalloc((void **) &d_odata, mem_size));
    cutilSafeCall( cudaMemcpy(d_odata, h_odata, mem_size, 
                              cudaMemcpyHostToDevice));
    //cutilSafeCall( cudaMemcpy2D(d_odata, pitched_sz, h_odata, standart_pitch, 
    //             standart_pitch, dim_len.y, cudaMemcpyHostToDevice));
                                
    
                                
    //cutilSafeCall( cudaMemcpyToArray( d_weights, 0, 0, h_weights, weight_size, cudaMemcpyHostToDevice) );

    cutilSafeCall( cudaMemcpy( d_rotability_even, h_rotability_even, rot_size,
                                cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_rotability_odd, h_rotability_odd, rot_size,
                                cudaMemcpyHostToDevice) );
                                
    
    cutilCheckMsg("Copy failed");
    // bind some data to textures
    //tex.addressMode[0] = cudaAddressModeClamp;
    //tex.addressMode[1] = cudaAddressModeClamp;
    //tex.filterMode = cudaFilterModePoint;
    //tex.normalized = false;    
    //  cudaBindTextureToArray(tex, d_idata, sys_channel_desc);
    
    //weights_tex.addressMode[0] = cudaAddressModeClamp;
    //weights_tex.addressMode[1] = cudaAddressModeClamp;
    //weights_tex.filterMode = cudaFilterModePoint;
    //weights_tex.normalized = false;    // access with integer texture coordinates
    //cudaBindTextureToArray(weights_tex, d_weights, w_channel_desc);
    char * filename = new char[256];

    if (params->save_bmp)
	{
		sprintf(filename, "res_%d.bmp", 0);
        save_bmp(h_idata, Coord(dim_len.x, dim_len.y, dim_len.z), filename);
    }

  #ifdef _MEM_DEBUG
    size_t err_sz = 6*sizeof(int);
    int * h_error = (int*) malloc(err_sz);
    memset(h_error, 0x00, err_sz);
    int * d_error = 0;
    cutilSafeCall( cudaMalloc((void **) &d_error, err_sz));
    cutilSafeCall( cudaMemcpy( d_error, h_error, err_sz, cudaMemcpyHostToDevice) );
  #endif
    for (int i = 0; i < params->max_iter; ++i)
    {
        //cudaBindTexture2D(0, &tex, d_idata, &sys_channel_desc, dim_len.x, dim_len.y, pitched_sz);
	#ifndef GPU_RAND
        generate_rnd(h_random, random_elements);
        cp_rnd_dev(h_random, d_random, random_mem_size);
	#endif
        
        cudaThreadSynchronize();
        cutilCheckMsg("Bind failed");
        // execute the kernel
	//#ifdef _DEBUG
        //cutilSafeCall(cudaMemset2D	(	d_odata, pitched_sz, 0xBB, standart_pitch, dim_len.x));
        cutilSafeCall(cudaMemset(d_odata, 0xBB, mem_size));
	//#endif
        caKernel<<< grid, threads >>>( d_idata, d_rotability_even, weights, 
      #ifdef _MEM_DEBUG
                                       d_error,
      #endif
                                       dim_len, 0, d_random, d_odata);
        
        // check if kernel execution generated and error
        cudaThreadSynchronize();
        cutilCheckMsg("Kernel execution failed");
        
    #ifdef _DBL_CP_DEBUG
        //cutilSafeCall(cudaMemcpy2D(h_odata, standart_pitch, d_odata, pitched_sz, 
        //     standart_pitch, dim_len.y, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
        cutilCheckMsg("Wrong copy d_odata");
    #endif
    #ifdef _MEM_DEBUG        
        cutilSafeCall(cudaMemcpy(h_error, d_error, err_sz, cudaMemcpyDeviceToHost));
        if (h_error[0] != 0)
        {
            fprintf (stderr, "On %d-th iteration, even step\n", i);
            fprintf (stderr, "Error: %d. (%d, %d, %d, %d, %d)\n", h_error[0], h_error[1], h_error[2], h_error[3], h_error[4], h_error[5]);
            return;
        }
	#endif
        
	#ifndef GPU_RAND
        generate_rnd(h_random, random_elements);
        cp_rnd_dev(h_random, d_random, random_mem_size);
	#endif
        // cudaBindTexture2D(0, &tex, d_odata, &sys_channel_desc, dim_len.x, dim_len.y, pitched_sz);
        cudaThreadSynchronize();
        cutilCheckMsg("Bind failed");
        
	//#ifdef _DEBUG
        //cutilSafeCall(cudaMemset2D	(	d_idata, pitched_sz, 0xBB, standart_pitch, dim_len.x));
        cutilSafeCall(cudaMemset(d_idata, 0xBB, mem_size));
	//#endif
        
        caKernel <<< grid, threads >>>( d_odata, d_rotability_odd, weights, 
      #ifdef _MEM_DEBUG
                                        d_error,
      #endif
                                        dim_len, 1, d_random, d_idata);
    
        // check if kernel execution generated and error
        cudaThreadSynchronize();
        cutilCheckMsg("Kernel execution failed");
    #ifdef _DBL_CP_DEBUG
        //cutilSafeCall(cudaMemcpy2D(h_odata, standart_pitch, d_idata, pitched_sz, 
        //     standart_pitch, dim_len.y, cudaMemcpyDeviceToHost));
        
        cutilSafeCall(cudaMemcpy(h_odata, d_idata, mem_size, cudaMemcpyDeviceToHost));
        cutilCheckMsg("Wrong copy d_idata");
    #endif
    #ifdef _MEM_DEBUG

        cutilSafeCall(cudaMemcpy(h_error, d_error, err_sz, cudaMemcpyDeviceToHost));
        if (h_error[0] != 0)
        {
            fprintf (stderr, "On %d-th iteration, odd step\n", i);
            fprintf (stderr, "Error: %d (%d, %d, %d, %d, %d)\n", h_error[0], h_error[1], h_error[2], h_error[3], h_error[4], h_error[5]);
            return;
        }

	#endif
        if ((i+1) % params->count_every == 0)
        {
                cutilSafeCall(cudaMemcpy(h_odata, d_idata, mem_size, cudaMemcpyDeviceToHost));
                cutilCheckMsg("Wrong copy d_idata");
		if (params->save_bmp)
		{
		    sprintf(filename, "res_%d.bmp", (i+1));
        	    save_bmp(h_odata, Coord(dim_len.x, dim_len.y, dim_len.z), filename);
        	}
        	int cnt = GetDrugCount(params, h_odata);
        	sprintf(str_buf, "%i\t%i\n", (i+1), cnt);
        	Log(params, str_buf);
        	printf("%s", str_buf);
			save_dump(params, h_odata);
        }
    }
	
    delete [] filename;

    // copy result from device to host
    //cutilSafeCall(cudaMemcpy2D(h_odata, standart_pitch, d_odata, pitched_sz, 
    //         standart_pitch, dim_len.y, cudaMemcpyDeviceToHost));
    //cutilSafeCall(cudaMemset(d_odata, 0xBB, mem_size));

#ifdef _DEBUG
    // check material ballance
    int check_in[LABEL_LAST];
    int check_out[LABEL_LAST];
    bool allOK = true;
    for (int i = 0; i < LABEL_LAST; ++i)
    {
      check_in[i] = 0;
      check_out[i] = 0;
    }
    for (int i = 0; i < dim_len.x*dim_len.y; ++i)
    {
        if (h_idata[i] >= LABEL_LAST)
        {
          printf("Input out of range!\n");
          allOK = false;
          break;
        }
        if (h_odata[i] >= LABEL_LAST)
        {
          printf("Output out of range!\n");
          allOK = false;
          break;
        }
        check_in[h_idata[i]] += 1;
        check_out[h_odata[i]] += 1;
    }
    for (int i = 0; i < LABEL_LAST; ++i)
    {
      printf("Material balance for %d (%d -> %d)!\n", i, check_in[i], check_out[i]);
      if (check_in[i] != check_out[i])
      {
        allOK = false;
      }
    }
    printf(allOK?"OK\n":"Ne OK\n");
    
//    printf("h_odata: \n");
//    for (int j = 0; j < dim_len.y; ++j)
//    {
//        for (int i = 0; i < dim_len.x; ++i)
//        {
//            printf("%d\t", h_odata[COORD_TO_ABS(i, j)]);
//        }
//        printf("\n");
//    }
#endif
             
    // cleanup memory
    //free( h_idata);
    free( h_odata);
    free( h_random);
    free( h_weights);
    free( h_rotability_even);
    free( h_rotability_odd);
//    free( str_buf);
    
    delete [] str_buf;
    
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_random));
    //cutilSafeCall(cudaFreeArray(d_weights));
    cutilSafeCall(cudaFree(d_rotability_even));
    cutilSafeCall(cudaFree(d_rotability_odd));
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();
}

CoordVec * GetNeightMap()
{
	CoordVec * res = new CoordVec;
    int * vals = new int[2*DIMS];
	for (int d = 0; d < DIMS; ++d)
	{
		vals[d] = 1;
		vals[d + DIMS] = -1;
	}
	for (int i = 0; i < 2*DIMS; ++i)
	{
		Coord c(0, 0, 0);
		c.SetCoord(i%DIMS, vals[i]);
		res->push_back(c);
	}
	return res;
}

int GetDrugCount(t_params * params, ElementType * fld)
{
	CoordVec * point_list = new CoordVec;
	CoordVec * neight_map = GetNeightMap();
	Coord up, down;
	up = get_upper_coord(params);
	down = get_lower_coord(params);
	dim3 dim_len = GetDims(params);
	int k = 0;
	for (int i = up.GetCoord(0); i < down.GetCoord(0); i++)
    {
    	for (int j = up.GetCoord(1); j < down.GetCoord(1); j++)
		{
		#if (DIMS == 3)
			for(k = up.GetCoord(2); k < down.GetCoord(2); k++)
			{
				ElementType lbl = fld[COORD_TO_ABS(i, j, k)];
		#else
			ElementType lbl = fld[COORD_TO_ABS(i, j)];
		#endif
			if (lbl == LABEL_AG)
            {
				CountAround(params, fld, neight_map, i, j, k, point_list);
			}
		#if (DIMS ==3)
			}
		#endif
        }
    }
	int res = point_list->size();
	delete point_list;
	delete neight_map;
	return res;
}

void CountAround(t_params * params, ElementType * fld, const CoordVec * neight_map, int x, int y, int z, CoordVec * point_list)
{
	dim3 dim_len = GetDims(params);
	for (int n = 0; n < neight_map->size(); ++n)
	{
		Coord curr_n = neight_map->at(n);
		int i = curr_n.GetCoord(0);
		int j = curr_n.GetCoord(1);
		int k = 0;
		ElementType lbl;
	#if (DIMS == 2)
		lbl = fld[COORD_TO_ABS(x+i, y+j)];
	#else
		k = curr_n.GetCoord(2);
		lbl = fld[COORD_TO_ABS(x+i, y+j, z+k)];
	#endif
		if (lbl == LABEL_DRUG)
		{
			Coord c(x + i, y + j, z + k);
			if (find(point_list->begin(), point_list->end(), c) == point_list->end())
			{
				point_list->push_back(c);
				CountAround(params, fld, neight_map, x + i, y + j, z + k, point_list);
			}
		}
	}
}

dim3 GetDims(t_params * params)
{
	dim3 res(params->n, params->n, params->n);
	return res;
}

void Log(t_params * params, char * str)
{
	FILE * pFile;
	pFile = fopen (params->print_to, "a");
	if (pFile != NULL)
	{
		fputs (str, pFile);
		fclose (pFile);
	}
}

void ClearLog(t_params * params)
{
	FILE * pFile;
	pFile = fopen (params->print_to, "w");
	fclose(pFile);
}

void PrintParams(t_params * params)
{
	char * str_buf = new char[512];
	sprintf(str_buf, "D_A = %f\n", params->D_A);
	Log(params, str_buf);
	sprintf(str_buf, "D_D = %f\n", params->D_D);
	Log(params, str_buf);
	sprintf(str_buf, "D_E = %f\n", params->D_E);
	Log(params, str_buf);
	sprintf(str_buf, "A_A = %f\n", params->A_A);
	Log(params, str_buf);
	sprintf(str_buf, "A_E = %f\n", params->A_E);
	Log(params, str_buf);
	sprintf(str_buf, "E_E = %f\n", params->E_E);
	Log(params, str_buf);
	
	sprintf(str_buf, "NeighCnt = %i\n", params->neigh_cnt);
	Log(params, str_buf);
	sprintf(str_buf, "IterCnt = %i\n", params->max_iter);
	Log(params, str_buf);
	sprintf(str_buf, "m = %i\n", params->m);
	Log(params, str_buf);
	sprintf(str_buf, "n = %i\n", params->n);
	Log(params, str_buf);
	sprintf(str_buf, "DrugCnt = %i\n", params->drug_cnt);
	Log(params, str_buf);
	
	sprintf(str_buf, "StructFile = %s\n", params->struct_filename);
	Log(params, str_buf);
	if (params->neg_filename != NULL)
		sprintf(str_buf, "NegativeFile = %s\n", params->neg_filename);
	else
		sprintf(str_buf, "NegativeFile = None\n");
	Log(params, str_buf);
	sprintf(str_buf, "LogFile = %s\n", params->print_to);
	Log(params, str_buf);
	sprintf(str_buf, "count_every = %i\n", params->count_every);
	Log(params, str_buf);
	sprintf(str_buf, "Device = %i\n", params->device);
	Log(params, str_buf);
	delete [] str_buf;
}
