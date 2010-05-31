/*
 * CellsField.h
 *
 *  Created on: 22.01.2010
 *      Author: andrey
 */

#ifndef CELLSFIELD_H_
#define CELLSFIELD_H_

#include "Coord.h"
#include "CommonFieldMagic.h"
#include "Field.h"

class CellsField // : public Field
{
public:
	CellsField();
	CellsField(const Coord &, const Coord &);
	~CellsField();

	FieldElement GetElement(const Coord &) const;
	bool IsSet(const Coord &) const;
	void SetElement(const Coord &);
	void UnSetElement(const Coord &);

	void SetElementVal(const Coord &, const FieldElement);
	FieldElement GetElementVal(const Coord &);

	void Clear();

	Coordinate GetTotalElements() const;
	Coordinate GetCellsCnt() const;

	Coord GetSize() const;
	Coord GetNull() const
	{
		return *mNullPnt;
	}
	size_t GetDims() const;

	void tofile(char * fileName) const;
	friend CellsField * fromfile(char * fileName);

	void Fill(FieldElement);

	void Resize(Coord & newSize, Coord & leftUpperCorner);

	bool IsElementInField(const Coord &) const;
	const FieldElement * GetCells()
	{
		return mCells;
	}
	void BindCells(FieldElement * new_cells)
	{
		delete [] mCells;
		mCells = new_cells;
	}
private:
	Coord * mSize;
	Coord * mNullPnt;
	FieldElement * mCells;
	size_t mDims;

protected:
	Coordinate CoordToAbs(const Coord &) const;

	void CreateCache(const Coord &);
};

#endif /* CELLSFIELD_H_ */
