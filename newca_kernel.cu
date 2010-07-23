/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* 
 * Device code.
 */

#ifndef _NEWCA_KERNEL_H_
#define _NEWCA_KERNEL_H_

#include "genca.h"
#include "rnd_gen.cu"


#ifdef  __DEVICE_EMULATION__
#include <stdio.h>
#endif

//texture<unsigned char, DIMS, cudaReadModeElementType> tex;
//texture<float, 2, cudaReadModeElementType> weights_tex;
//__device__ __constant__ float weights[LABEL_LAST*LABEL_LAST];
//__device__ float weights[LABEL_LAST*LABEL_LAST];

#if (DIMS == 2)

#if (NEIGHS == 5)
__device__ __constant__ signed char neighbours_deltas[] =  {
    -1, 1, -1, 0, -1, -1, 0, -1, 1, -1, 
    0, -1, 1, -1, 2, -1, 2, 0, 2, 1, 
    -1, 0, -1, 1, -1, 2, 0, 2, 1, 2, 
    0, 2, 1, 2, 2, 2, 2, 1, 2, 0
};
// [0, 0]: [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]
// -1, 1, -1, 0, -1, -1, 0, -1, 1, -1,
// [1, 0]: [0, -1], [1, -1], [2, -1], [2, 0], [2, 1]
// 0, -1, 1, -1, 2, -1, 2, 0, 2, 1,
// [0, 1]: [-1, 0], [-1, 1], [-1, 2], [0, 2], [1, 2]
// -1, 0, -1, 1, -1, 2, 0, 2, 1, 2, 
// [1, 1]: [0, 2], [1, 2], [2, 2], [2, 1], [2, 0]
// 0, 2, 1, 2, 2, 2, 2, 1, 2, 0
//};

#else
#if (NEIGHS == 2)
__device__ __constant__ signed char neighbours_deltas[] =  {
    -1, 0, 0, -1,
    1, -1, 2, 0, 
    -1, 1, 0, 2, 
    1, 2, 2, 1
};
#endif
#endif


#define ROTATIONS 3
#define STAY_ROTATION 0

__device__ __constant__ signed char rotations[] =  {
    0, 0, 1, 0, 0, 1, 1, 1,
    1, 0, 1, 1, 0, 0, 0, 1, 
    0, 1, 0, 0, 1, 1, 1, 0
};
    // 0 rotation:     [0, 0], [1, 0], [0, 1], [1, 1]
    // 0, 0, 1, 0, 0, 1, 1, 1,
    // left rotation:  [1, 0], [1, 1], [0, 0], [0, 1]
    // 1, 0, 1, 1, 0, 0, 0, 1,
    // right rotation: [0, 1], [0, 0], [1, 1], [1, 0]
    //0, 1, 0, 0, 1, 1, 1, 0
//};


#else

__device__ __constant__ signed short neighbours_deltas[] = 
{
// [0, 0, 0]: [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]
-1, 1, -1, 0, -1, -1, 0, -1, 1, -1,
// [1, 0, 0]: [0, -1], [1, -1], [2, -1], [2, 0], [2, 1]
0, -1, 1, -1, 2, -1, 2, 0, 2, 1,
// [0, 1, 0]: [-1, 0], [-1, 1], [-1, 2], [0, 2], [1, 2]
-1, 0, -1, 1, -1, 2, 0, 2, 1, 2, 
// [1, 1, 0]: [0, 2], [1, 2], [2, 2], [2, 1], [2, 0]
0, 2, 1, 2, 2, 2, 2, 1, 2, 0
};


#endif


__device__ int
chooseRot(float * energies, 
#ifdef _MEM_DEBUG
int * error, 
#endif
float rnd)
{
  float energ_sum=0.0f;
  for (int rot = 0; rot < ROTATIONS; rot++)
  {
      energies[rot] = exp(energies[rot]);
      energ_sum+= energies[rot];
  }
            
  #ifdef __DEVICE_EMULATION__
  printf("Energies: ");
  for (int rot = 0; rot < ROTATIONS; rot++)
  {
      printf("%f\t", energies[rot]);
  }
  printf("\nenerg_sum = %f\n", energ_sum);
  printf("Rand: %f\n", rnd);
  #endif
  for (int rot = 0; rot < ROTATIONS; rot++)
  {
        energies[rot] /= energ_sum;
  }
  
  // FIXME: unroll it!
  float curr_sum = energies[0];
  int res = 0;
  while(curr_sum < rnd && res < ROTATIONS)
  {
     #ifdef _MEM_DEBUG
        if (!(0 <= res && res < ROTATIONS)) {
           error[0] = 153;
           return ROTATIONS-1;
        }
     #endif 
      res++;
      curr_sum += energies[res];
  }
  return res;
}

////////////////////////////////////////////////////////////////////////////////
//! Rotation procedure
//! @param g_rotability data of blocks rotations
//! @param g_rand - random numbers array
//! @param g_nsys - new system state
////////////////////////////////////////////////////////////////////////////////
#if (DIMS == 2)
__global__ void
//caKernel( ElementType* g_insys, RotationType * g_rotability, dim3 dim_len, int odd, float* g_rand, ElementType* g_nsys, size_t sys_pitch) 
caKernel( const ElementType* g_insys, const RotationType * g_rotability, const float * weights,
#ifdef _MEM_DEBUG
int * error,
#endif
const dim3 dim_len, const int odd, const RandomType* g_rand, RandomType * g_new_rand, ElementType* g_nsys) 
{
  // 0. Compute block variables: number et al
  int topX,topY;
  topX = (blockIdx.x*blockDim.x + threadIdx.x);
  topY = (blockIdx.y*blockDim.y + threadIdx.y);
  
  int block_num = topX + topY*dim_len.x/2;
  topX = topX * 2 + odd;
  topY = topY * 2 + odd;
  
  if (topX >= dim_len.x || topY >= dim_len.y)
  {
     return;
  }
  int curr_ind;
  #ifdef _MEM_DEBUG
  int max_size = dim_len.x*dim_len.y;
  error[0] = 0;
  if (block_num < 0 || block_num >= max_size/4) {
     error[0] = 194;
     return; 
  }
  #endif
  
  #ifdef __DEVICE_EMULATION__
  printf("Block %d\n", block_num);
  #endif
  
  #ifdef __DEVICE_EMULATION__
  printf("Input:\n");
  
  for (int dy = 0; dy < 2; ++dy)
  {
      for (int dx = 0; dx < 2; ++dx)
      {
        printf("%d\t", g_insys[COORD_TO_ABS(topX+dx, topY+dy)]);
      }
      printf("\n");
  }
  printf("Output:\n");
  for (int dy = 0; dy < 2; ++dy)
  {
      for (int dx = 0; dx < 2; ++dx)
      {
          printf("%d\t", g_nsys[COORD_TO_ABS(topX+dx, topY+dy)]);
      }
      printf("\n");
  }
  #endif
  
  // 1. check if that block is rotatable:
  int choosen_rot = STAY_ROTATION;
  if (g_rotability[block_num] != 0)
  {
      
      #ifdef __DEVICE_EMULATION__
    printf("Rotate it\n");
    #endif
      
    // 2. collect energy for all rotations
    int curr_rot = 0;
    float energies[ROTATIONS];
    // #pragma unroll
    if (topX == 0 || topY == 0 || topX >= dim_len.x-2 || topY >= dim_len.y-2)
    {
      #ifdef __DEVICE_EMULATION__
      printf("External\n");
      #endif
      
      for (int rot = 0; rot < ROTATIONS; rot ++)
      {
          energies[rot] = 0.0f;
        
        int curr_neigh = 0;
      
        // #pragma unroll
        for (int i = 0; i < 4; i++)
        {
          int elx = (topX+rotations[curr_rot]);
          if (elx >= dim_len.x) elx -= dim_len.x;
          int ely = (topY+rotations[curr_rot+1]);
          if (ely >= dim_len.y) elx -= dim_len.y;
          curr_ind = COORD_TO_ABS(elx, ely);
          #ifdef _MEM_DEBUG
          if (curr_ind < 0 || curr_ind >= max_size) {
              error[0] = 260; error[1] = elx; error[2] = ely; error[3] = curr_rot; error[4] = rotations[curr_rot], error[5] = rotations[curr_rot+1];
              return;
          }
          #endif 
          int element = g_insys[curr_ind];
          
          curr_rot += 2;
          
          // #pragma unroll 5
          for (int neigh = 0; neigh < NEIGHS; neigh++)
          { 
              int nx = (topX+neighbours_deltas[curr_neigh]);
              if (nx < 0) nx += dim_len.x;
              else if (nx >= dim_len.x) nx -= dim_len.x;
              
              int ny = (topY+neighbours_deltas[curr_neigh+1]);
              if (ny < 0) ny += dim_len.y;
              else if (ny >= dim_len.x) ny -= dim_len.y;
              
              curr_ind = COORD_TO_ABS(nx, ny);
           #ifdef _MEM_DEBUG
              if (curr_ind < 0 || curr_ind >= max_size) {
                    error[0] = 282;
                    return; 
              }
           #endif
              int neight_val = (int)g_insys[curr_ind];
            
              curr_ind = element + LABEL_LAST * neight_val;
           #ifdef _MEM_DEBUG
              if (curr_ind < 0 || curr_ind >= LABEL_LAST*LABEL_LAST) {
                error[0] = 298; error[1] = nx; error[2] = ny; error[3] = curr_neigh; error[4] = neighbours_deltas[curr_neigh]; error[5] = neighbours_deltas[curr_neigh+1];
                return;
              }
           #endif
              curr_neigh+=2;
              energies[rot] += weights[curr_ind];
          }
        }
      }
    }
    else
    {
      #ifdef __DEVICE_EMULATION__
      printf("Internal\n");
      #endif
      
      for (int rot = 0; rot < ROTATIONS; rot ++)
      {
          energies[rot] = 0.0f;
        int curr_neigh = 0;
      
        for (int i = 0; i < 4; i++)
        {
          curr_ind = COORD_TO_ABS((topX+rotations[curr_rot]), (topY+rotations[curr_rot+1]));
          #ifdef _MEM_DEBUG
          if (curr_ind < 0 || curr_ind >= max_size) {
              error[0] = 313; 
              return;
          }
          #endif
          int element = g_insys[curr_ind];
          curr_rot += 2;
          
          // #pragma unroll 5
          for (int neigh = 0; neigh < NEIGHS; neigh++)
          {
              int nx = (topX+neighbours_deltas[curr_neigh]);
              int ny = (topY+neighbours_deltas[curr_neigh+1]);
              
              curr_ind = COORD_TO_ABS(nx, ny);
              #ifdef _MEM_DEBUG
              if (curr_ind < 0 || curr_ind >= max_size) {
                  error[0] = 329;
                  return;
              }
              #endif
              int neight_val = (int)g_insys[curr_ind];
              
              curr_neigh+=2;
              curr_ind = element + LABEL_LAST * neight_val;
              #ifdef _MEM_DEBUG
              if (curr_ind < 0 || curr_ind >= LABEL_LAST*LABEL_LAST) {
                 error[0] = 339;
                 return;
              }
              #endif
              energies[rot] += weights[curr_ind];
          }
        }
      }
    }
    
    // 3. choose random rotation
    #ifndef GPU_RAND
    RandomType rnd = g_rand[block_num];
    #else
    RandomType rnd = randGPU(block_num);
    //g_new_rand[block_num] = randGPU(rnd);
    #endif
    choosen_rot = chooseRot(energies, 
#ifdef _MEM_DEBUG
                            error, 
#endif
                            ((float)rnd+1.0f)/MY_RAND_MAX);
    
    #ifdef __DEVICE_EMULATION__
    printf("Choosen %d\n", choosen_rot);
    printf("Old rand: %u, New rand: %u\n", rnd, g_rand[block_num]);
    #endif
      
  }
  // 4. Execute rotation
  int curr_neigh = choosen_rot*DIMS*2*2;
  
  for (int dy = 0; dy < 2; dy++)
  {
      for (int dx = 0; dx < 2; dx++)
      {
          int dest_x = topX + dx;
          if (dest_x >= dim_len.x) dest_x -= dim_len.x;
          int dest_y = topY + dy;
          if (dest_y >= dim_len.y) dest_y -= dim_len.y;
          //ElementType * sys_row = (ElementType*)((char*)g_nsys+dest_y*sys_pitch);
          
          int src_x = topX+rotations[curr_neigh];
          if (src_x >= dim_len.x) src_x -= dim_len.x;
          int src_y = topY+rotations[curr_neigh+1];
          if (src_y >= dim_len.y) src_y -= dim_len.y;
          
          #ifdef __DEVICE_EMULATION__
          
        printf("(%d %d) %d -> (%d %d) %d", src_x, src_y, g_insys[COORD_TO_ABS(src_x, src_y)] , 
               dest_x, dest_y, g_nsys[COORD_TO_ABS(dest_x, dest_y)]);
        #endif
          
          //sys_row[dest_x] = tex2D(tex, src_x, src_y);
          #ifdef _MEM_DEBUG
          curr_ind = COORD_TO_ABS(dest_x, dest_y);
          if (curr_ind < 0 || curr_ind >= max_size) {
              error[0]=404;
              return;
          }
          curr_ind = COORD_TO_ABS(src_x, src_y);
          if (curr_ind < 0 || curr_ind >= max_size) {
              error[0] = 409;
              return;
          }
          #endif
          g_nsys[COORD_TO_ABS(dest_x, dest_y)] = g_insys[COORD_TO_ABS(src_x, src_y)];
          
          #ifdef __DEVICE_EMULATION__
        printf(": %d\n", g_nsys[COORD_TO_ABS(dest_x, dest_y)]);
        #endif
          
          curr_neigh+=2;
      }
  }
}

//__global__ void
//caKernelOdd( RotationType * g_rotability, dim3 dim_len, float* g_rand, ElementType* g_nsys) 
//{
//  // 0. Compute block variables: number et al
//  int topX,topY;
//  topX = (blockIdx.x*blockDim.x + threadIdx.x);
//  topY = (blockIdx.y*blockDim.y + threadIdx.y);
//  
//  int block_num = topX + topY*dim_len.x/2;
//  topX = topX * 2 + 1;
//  topY = topY * 2 + 1;
//  
//  if (topX >= dim_len.x || topY >= dim_len.y)
//  {
//     return;
//  }
//  // 1. check if that block is rotatable:
//  int choosen_rot = 0;
//  if (g_rotability[block_num] != 0)
//  {
//    // 2. collect energy for all rotations
//    int curr_rot = 0;
//    float energies[ROTATIONS];
////    #pragma unroll
//    if (topX == dim_len.x-1 || topY == dim_len.y-1)
//      for (int rot = 0; rot < ROTATIONS; rot ++)
//      {
//          energies[rot] = 0.0f;
//        
//        int curr_neigh = 0;
//      
////        #pragma unroll
//        for (int i = 0; i < 4; i++)
//        {
//          int ex = (topX+rotations[curr_rot]);
//          if (ex >= dim_len.x) ex -= dim_len.x;
//          
//          int ey = (topY+rotations[curr_rot+1]);
//          if (ey >= dim_len.y) ey -= dim_len.y;
//          
//          int element = tex2D(tex, ex, ey);
//          
//          curr_rot += 2;
//          
//          // #pragma unroll 5
//          for (int neigh = 0; neigh < NEIGHS; neigh++)
//          {
//              int nx = (topX+neighbours_deltas[curr_neigh]);
//              if (nx >= dim_len.x) nx -= dim_len.x;
//              
//              int ny = (topY+neighbours_deltas[curr_neigh+1]);
//              if (ny >= dim_len.x) ny -= dim_len.y;
//              
//              int neight_val = (int)tex2D(tex, nx, ny);
//            
//            curr_neigh+=2;
//            energies[rot] += tex2D(weights_tex, element, neight_val);
//          }
//        }
//      }
//    else
//      for (int rot = 0; rot < ROTATIONS; rot ++)
//      {
//          energies[rot] = 0.0f;
//        int curr_neigh = 0;
//      
//        for (int i = 0; i < 4; i++)
//        {
//          int element = tex2D(tex, (topX+rotations[curr_rot]), 
//            (topY+rotations[curr_rot+1]));
//          curr_rot += 2;
//          
//          // #pragma unroll 5
//          for (int neigh = 0; neigh < NEIGHS; neigh++)
//          {
//              int nx = (topX+neighbours_deltas[curr_neigh]);
//              int ny = (topY+neighbours_deltas[curr_neigh+1]);
//              
//              int neight_val = (int)tex2D(tex, nx, ny);
//            
//            curr_neigh+=2;
//            energies[rot] += tex2D(weights_tex, element, neight_val);
//          }
//        }
//    }
//    
//    // 3. choose random rotation
//    choosen_rot = chooseRot(energies, g_rand[block_num]);
//  }
//  // 4. Execute rotation
//  int curr_neigh = choosen_rot*DIMS*2*2;
//  
//  if (topX == dim_len.x-1 || topY == dim_len.y-1)
//  {
//      for (int dy = 0; dy < 2; dy++)
//    {
//        for (int dx = 0; dx < 2; dx++)
//        {
//            int src_e_x = topX+rotations[curr_neigh];
//            if (src_e_x >= dim_len.x) src_e_x -= dim_len.x;
//            int src_e_y = topY+rotations[curr_neigh+1];
//            if (src_e_y >= dim_len.y) src_e_y -= dim_len.y;
//            int dest_e_x = topX + dx;
//            if (dest_e_x >= dim_len.x) dest_e_x -= dim_len.x;
//            int dest_e_y = topY + dy;
//            if (dest_e_y >= dim_len.y) dest_e_y -= dim_len.y;
//            
//            g_nsys[COORD_TO_ABS(dest_e_x, dest_e_y)] = tex2D(tex, src_e_x, src_e_y);
//            
//            curr_neigh += 2;
//        }
//    }
//  }
//  else
//  {
//    for (int dy = 0; dy < 2; dy++)
//    {
//        for (int dx = 0; dx < 2; dx++)
//        {
//            g_nsys[COORD_TO_ABS(topX + dx, topY+dy)] = tex2D(tex, (topX+rotations[curr_neigh]),
//             (topY+rotations[curr_neigh+1]));
//          curr_neigh+=2;
//        }
//    }
//  }
//}

#else
__global__ void
caKernel(unsigned char* g_rotability, dim3 dim_len, float* g_rand, int blockIdz, ElementType* g_nsys)
{
  // 0. Compute block variables: number et al
  int topX, topY, topZ;
  topX = (blockIdx.x*blockDim.x + threadIdx.x);
  topY = (blockIdx.y*blockDim.y + threadIdx.y);
  topZ = (blockIdz * blockDim.z + threadIdx.z);
  
  // ---------------------------------------------
  
}
#endif
#endif // #ifndef _NEWCA_KERNEL_H_
