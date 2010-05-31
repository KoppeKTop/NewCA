#ifndef _FIELD_SAVER_H_
#define _FIELD_SAVER_H_

// Operation to save the field
#include "CellsField.h"
#include "genca.h"

extern "C" int save_bmp(const ElementType * fld, const Coord sz, const char * filename);

#endif
