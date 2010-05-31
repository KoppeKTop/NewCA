#ifndef _LOAD_PARAMS_H_
#define _LOAD_PARAMS_H_

#include "genca.h"
#include "newca.h"
#include "Coord.h"
#include "CellsField.h"

extern "C" int get_params(t_params * params, char * filename);
extern "C" int fill_with_structure(t_params * params, CellsField * fld);
extern "C" int fill_with_drug(t_params * params, CellsField * fld);
extern "C" bool file_exists(const char * filename);
extern "C" CoordVec * load_struct (char * filename);
extern "C" void get_rotation_maps(CellsField * fld, RotationType * even, RotationType * odd);
extern "C" Coord get_upper_coord(const t_params * params);
extern "C" Coord get_lower_coord(const t_params * params);

int check_params(t_params * params, int ret);

#endif
