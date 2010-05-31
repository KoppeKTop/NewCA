/*
 *  dumping.h
 *  
 *
 *  Created by Andrey on 27/5/10.
 *  Copyright 2010 MUCTR. All rights reserved.
 *
 */

#ifndef _DUMPING_H_
#define _DUMPING_H_
#include "newca.h"

extern "C" void save_dump(const t_params * params, const ElementType * src);
extern "C" int load_dump(const t_params * params, ElementType * dst);

#endif
