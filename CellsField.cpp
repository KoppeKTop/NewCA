/*
 * CellsField.cpp
 *
 *  Created on: 22.01.2010
 *      Author: andrey
 */

#include "CellsField.h"

CellsField::CellsField()
{
    mNullPnt = new Coord;
    mSize = new Coord;
}

CellsField::CellsField(const Coord & size, const Coord & UpperCorner)
		// where size - size of field
		// UpperCorner - upper left corner coordinate
{
	mNullPnt = new Coord;
	*mNullPnt = UpperCorner;
	mSize = new Coord;
	*mSize = size;
	size_t total = GetElementsFromSize(size);
	mCells = new FieldElement[total];
}

CellsField::~CellsField()
{
	delete [] mCells;
	delete mSize;
	delete mNullPnt;
}

FieldElement CellsField::GetElement(const Coord & c) const
{
	// relative coordinate
	if (IsElementInField(c))
	{
		Coordinate absCoord = CoordToAbs(c);
		FieldElement res = mCells[absCoord];
		return res;
	}
	throw OutOfBoundError();
}

bool CellsField::IsSet(const Coord & c) const
{
	FieldElement curr = GetElement(c);
	bool res = (curr == OCUPIED_CELL);
	return res;
}

void CellsField::SetElementVal(const Coord & c, FieldElement val)
{
	if (IsElementInField(c))
	{
		Coordinate absCoord = CoordToAbs(c);
		mCells[absCoord] = val;
		return;
	}
	throw OutOfBoundError();
}

FieldElement CellsField::GetElementVal(const Coord & c)
{
	if (IsElementInField(c))
	{
		Coordinate absCoord = CoordToAbs(c);
		return mCells[absCoord];
	}
	throw OutOfBoundError();
}

void CellsField::SetElement(const Coord & c)
{
	SetElementVal(c, OCUPIED_CELL);
}


void CellsField::UnSetElement(const Coord & c)
{
	SetElementVal(c, FREE_CELL);
}

void CellsField::Clear()
{
	Coordinate total = GetElementsFromSize(GetSize());
	for (int i = 0; i < total; i++)
	{
		mCells[total] = FREE_CELL;
	}
}

Coordinate CellsField::GetTotalElements() const
{
	Coordinate total = GetElementsFromSize(GetSize());
	return total;
}

Coord CellsField::GetSize() const
{
	return * mSize;
}

Coordinate CellsField::CoordToAbs(const Coord & c) const
{
	Coord correctedC = c - GetNull();
	Coordinate res = 0;
	int sizeMul = 1;
	// result == X + Y * MaxX + Z * MaxX * MaxY
	for (size_t i=0; i < Coord::GetDefDims(); i++)
	{
		res += correctedC.GetCoord(i) * sizeMul;
		sizeMul *= GetSize().GetCoord(i);
	}
	return res;
}

bool CellsField::IsElementInField(const Coord & c) const
{
	bool res = true;
	for (size_t i = 0; i < Coord::GetDefDims(); i++)
	{
		Coordinate currCord = c.GetCoord(i);
		Coordinate leftC = GetNull().GetCoord(i);
		Coordinate rigthC = leftC + GetSize().GetCoord(i);
		if (currCord < leftC || rigthC <= currCord) {
			res = false;
			break;
		}
	}
	return res;
}

size_t CellsField::GetDims() const
{
	return mDims;
}


void CellsField::Fill(FieldElement val)
{
	Coordinate total = GetTotalElements();
	for (int pnt = 0; pnt < total; pnt++)
	{
		this->mCells[pnt] = val;
	}
}

Coordinate CellsField::GetCellsCnt() const
		// returns total amount of cells in field
{
	return GetElementsFromSize(GetSize());
}

void CellsField::tofile(char * fileName) const
{
	FILE * saveFile = fopen(fileName, "wb+");
	Coordinate total = GetElementsFromSize(GetSize());
	fwrite(this->mCells, sizeof(FieldElement), total, saveFile);
	fclose(saveFile);
}

void CellsField::Resize(Coord & newSize, Coord & leftUpperCorner)
{
	Coord oldSize = this->GetSize();
	if (oldSize == newSize)
	{
		this->Clear();
	}
	else
	{
		size_t newTotal = GetElementsFromSize(newSize);
		FieldElement * newCells = new FieldElement[newTotal];
		delete [] this->mCells;
		this->mCells = newCells;

		*mSize = newSize;
	}
	*mNullPnt = leftUpperCorner;
}

//void Expand(Coordinate);
//void Expand(Coordinate, Coordinate, Coordinate=0);

//CellsField * fromfile(char * fileName)
//{
//	// TODO write this function!
//	FILE * loadFile = fopen(fileName, "rb+");
//
//	// Define file size:
//	fseek(loadFile, 0L, SEEK_END);
//	long sz = ftell(loadFile);
//	fseek(loadFile, 0L, SEEK_SET);
//	long total = sz/sizeof(FieldElement);
//
//	Coordinate * cells = new Coordinate[total];
//	fread(cells, sizeof(FieldElement), total, loadFile);
//	fclose(loadFile);
//
//	delete [] cells;
//}
