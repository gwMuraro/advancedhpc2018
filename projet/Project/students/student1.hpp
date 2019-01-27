#pragma once

#include "ppm.hpp"

float student1(const PPMBitmap& in, PPMBitmap& out, const int size);

// You may export your own class or functions to communicate data between the exercises ...

class VectorizePPM {

	public : 
		// constructor
		VectorizePPM(PPMBitmap *bitmap) : bitmap (bitmap) {} ;
		
		// vectorizor 
		uchar3 * vectorize() const ;
	
		// accessors 
		PPMBitmap * getBitmap() const {return bitmap ;}
	
	private : 
		PPMBitmap * bitmap ;

} ;
