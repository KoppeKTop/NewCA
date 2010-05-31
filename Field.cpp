/*
 * Field.cpp
 *
 *  Created on: 11.01.2010
 *      Author: andrey
 */

#include "Field.h"

size_t GetElementsFromSize(const Coord & size)
{
	size_t total = 1;
	for (size_t i=0; i < Coord::GetDefDims(); i++)
	{
		total *= size.GetCoord(i);
	}
	return total;
}
