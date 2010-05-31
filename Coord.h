/*
 * Coord.h
 *
 *  Created on: 12.01.2010
 *      Author: andrey
 */

#ifndef COORD_H_
#define COORD_H_

//#include "CoordinateVector.h"

#include <vector>
#include <exception>
#include <fstream>
//#include "SortedVector.h"
//#include <boost/thread/mutex.hpp>
using namespace std;

typedef int Coordinate;
#define MAX_DIMS 3

class OutOfBoundError: public exception
{
  virtual const char* what() const throw()
  {
    return "Coordinate index out of bounds";
  }
};

//typedef Coordinate CoordinateVector[MAX_DIMS];

class no_mutex {
	public:
	void lock()
	{}
	void unlock()
	{}
};

class Coord {
public:
	Coord(Coordinate *);
	Coord(const Coord &);
	Coord(Coordinate=0, Coordinate=0, Coordinate=0);

	virtual ~Coord();

	Coordinate GetCoord(size_t) const;
	void SetCoord(size_t, Coordinate);
	void SetPosition(Coordinate, Coordinate, Coordinate=0);

	Coord operator+ (const Coord &) const;
	Coord operator% (const Coord &) const;
	Coord operator- (const Coord &) const;

	Coord operator+ (const Coordinate) const;
	Coord operator% (const Coordinate) const;
	Coord operator- (const Coordinate) const;
	Coord operator/ (const Coordinate) const;

	bool operator== (const Coord &) const;
	bool operator< (const Coord &) const;
	bool operator<= (const Coord &) const;
	bool operator> (const Coord &) const;
	bool operator>= (const Coord &) const;
	bool operator!= (const Coord &) const;

	Coord & operator= (const Coord & rhs);

	static size_t GetDefDims()	{	return	mDefDims;	}
	static void SetDefDims(size_t dims)
	{
	  //instanceLock.lock();
		if (instances != 0)
		{
		  // instanceLock.unlock();
			return;
		}
		if (2 <= dims || dims <= 3)
			mDefDims = dims;
		// instanceLock.unlock();
	}

	friend ostream& operator <<(ostream& stream, const Coord& temp);
private:
	Coordinate mCV[MAX_DIMS];
	size_t mDims;

	bool CheckBounds(size_t) const;

	static size_t mDefDims;
	static int instances;
	static no_mutex instanceLock;
};

typedef vector<Coord> CoordVec;
//typedef sorted_vector<Coord> SortedCoordVec;

#endif /* COORD_H_ */
