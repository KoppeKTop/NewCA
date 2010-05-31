/*
 * Coord.cpp
 *
 *  Created on: 12.01.2010
 *      Author: andrey
 */

#include "Coord.h"
#include <iostream>
using namespace std;

Coord::Coord(Coordinate X, Coordinate Y, Coordinate Z)
{
  //	Coord::instanceLock.lock();
	Coord::instances += 1;
       	// Coord::instanceLock.unlock();

	mDims = Coord::mDefDims;

	SetPosition(X,Y,Z);
}

Coord::Coord(Coordinate * cv)
{

  // Coord::instanceLock.lock();
	Coord::instances += 1;
	// Coord::instanceLock.unlock();

	mDims = Coord::mDefDims;

	for (size_t i=0; i < Coord::mDefDims; i++)
	{
		mCV[i] = cv[i];
	}
}

Coord::Coord(const Coord & c)
{

  // Coord::instanceLock.lock();
	Coord::instances += 1;
	// Coord::instanceLock.unlock();

	mDims = Coord::mDefDims;

	//mDims = c.mDims;
	for (size_t i=0; i < Coord::mDefDims; i++)
	{
		mCV[i] = c.GetCoord(i);
	}
}

Coord::~Coord()
{
  //Coord::instanceLock.lock();
	Coord::instances -= 1;
	// Coord::instanceLock.unlock();
};

bool Coord::CheckBounds(size_t index) const
{
	return (index < Coord::mDefDims);
}

Coordinate Coord::GetCoord(size_t num) const
{
	if (this->CheckBounds(num)) {
		return mCV[num];
	}
	throw OutOfBoundError();
}

void Coord::SetCoord(size_t num, Coordinate val)
{
	if (this->CheckBounds(num)) {
		this->mCV[num] = val;
		return;
	}
	throw OutOfBoundError();
}


void Coord::SetPosition(Coordinate X, Coordinate Y, Coordinate Z)
{
	mCV[0] = X;
	mCV[1] = Y;
	mCV[2] = Z;
}

Coord Coord::operator+ (const Coord &rhs) const
{
	Coord res;
	for (size_t i = 0; i < Coord::mDefDims; ++i)
		res.SetCoord(i, this->GetCoord(i) + rhs.GetCoord(i));
	return res;
}

Coord Coord::operator% (const Coord & rhs) const
{
	Coord res;
	for (size_t i = 0; i < Coord::mDefDims; ++i)
		res.SetCoord(i, this->GetCoord(i) % ((rhs.GetCoord(i) != 0)?rhs.GetCoord(i):1));
	return res;
}

Coord Coord::operator- (const Coord & rhs) const
{
	Coord res;
	for (size_t i = 0; i < Coord::mDefDims; ++i)
		res.SetCoord(i, this->GetCoord(i) - rhs.GetCoord(i));
	return res;
}

Coord Coord::operator/ (const Coordinate divide) const
{
	Coord res(*this);

	for (size_t i = 0; i < Coord::mDefDims; ++i)
		res.SetCoord(i, this->GetCoord(i) / divide);

	return res;
}

Coord Coord::operator+ (const Coordinate rhs) const
{
	Coord tmp(rhs,rhs,rhs);
	Coord res = *this + tmp;
	return res;
}

Coord Coord::operator% (const Coordinate rhs) const
{
	Coord tmp(rhs,rhs,rhs);
	Coord res = *this % tmp;
	return res;
}

Coord Coord::operator- (const Coordinate rhs) const
{
	Coord tmp(rhs,rhs,rhs);
	Coord res = *this - tmp;
	return res;
}

bool Coord::operator== (const Coord & rhs) const
{
	if (mDims != rhs.mDims)	return false;
	bool res = true;
	for (size_t i = 0; (i < Coord::mDefDims) && res; ++i)
		res = res && (rhs.GetCoord(i) == this->GetCoord(i));
	return res;
}

bool Coord::operator< (const Coord & rhs) const
{
	bool res = false;
	for (size_t i = 0; i < Coord::mDefDims; ++i)
	{
		if (this->GetCoord(i) < rhs.GetCoord(i)) {
			// less case - return true
			res = true;
			break;
		}
		if (this->GetCoord(i) == rhs.GetCoord(i))
			// equal case - continue check
			continue;
		// greater case - return false
		break;
	}
	return res;
}

bool Coord::operator<= (const Coord & rhs) const
{
	bool res = (*this < rhs || *this == rhs);
	return res;
}

bool Coord::operator> (const Coord & rhs) const
{
	bool res = !(*this <= rhs);
	return res;
}

bool Coord::operator>= (const Coord & rhs) const
{
	bool res = !(*this < rhs);
	return res;
}

bool Coord::operator!= (const Coord & rhs) const
{
	bool res = !(*this == rhs);
	return res;
}

Coord & Coord::operator= (const Coord & rhs)
{
	if (this != &rhs)
	{
		for (size_t i = 0; i<Coord::mDefDims; i++)
		{
			this->mCV[i] = rhs.mCV[i];
		}
	}
	return *this;
}

size_t Coord::mDefDims = 2;
int Coord::instances = 0;
// boost::mutex Coord::instanceLock;

ostream& operator <<(ostream& stream, const Coord& c)
{
	//stream << "(";
	stream << c.GetCoord(0);
	for (size_t d = 1; d < Coord::mDefDims; d++)
	{
		stream << "\t" << c.GetCoord(d);
	}
	//stream << ")";
	return stream;
}
