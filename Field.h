/*
 * Field.h
 *
 *  Created on: 11.01.2010
 *      Author: andrey
 */

#ifndef FIELD_H_
#define FIELD_H_

#include "Coord.h"
#include "CommonFieldMagic.h"
//#include <boost/thread/mutex.hpp>

class Field{
	// interface to all field classes
	// So! Field class. It will provide ability to many threads work with
	// number of points
public:
	virtual ~Field()=0;

	virtual FieldElement GetElement(const Coord & c) const = 0;
	virtual bool IsSet(const Coord &) const;
	virtual void SetElement(const Coord & c)=0;
	virtual void UnSetElement(const Coord & c)=0;

	virtual void Clear()=0;

	virtual void tofile(char * fileName) const;
};

size_t GetElementsFromSize(const Coord &);

#endif /* FIELD_H_ */
