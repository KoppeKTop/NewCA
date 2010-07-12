#ifndef _GENCA_H_
#define _GENCA_H_

#include "CommonFieldMagic.h"

#define DIMS 2

#define sqr(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))

#if (DIMS == 2)
#define COORD_TO_ABS(curr_x, curr_y) ((curr_x)+(curr_y)*dim_len.x)
#else
#define COORD_TO_ABS(curr_x, curr_y, curr_z) ((curr_x)+(curr_y)*dim_len.x+(curr_z)*dim_len.x*dim_len.y)
#endif

enum Labels {LABEL_EMPTY, LABEL_AG, LABEL_DRUG, LABEL_LAST};
enum FillType {FILL_ALL, FILL_INTERNAL, FILL_EXTERNAL };

typedef FieldElement ElementType;
typedef unsigned char RotationType;
typedef unsigned int RandomType;

#define NEIGHS 2

//#define GPU_RAND

//#define _MEM_DEBUG
//#define M_DEBUG
//#define VERBOSE
//#define DUMP_ALL

#endif // #ifndef _GENCA_H_
