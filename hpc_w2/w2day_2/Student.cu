#include "D_Matrix.cuh"
#include "H_Matrix.cuh"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

//////////////////////////////////////////////////////////////////////////////////
// Exercice 1
bool D_Matrix::Exo1IsDone() {
	return true;
}

D_Matrix D_Matrix::operator+(const D_Matrix& that) const
{
	// do "d_val + that.d_val" 
	// creating a matrix 
	D_Matrix result(m_n); 
	
	// Adding the 2 matrixs
	const int size = m_n * m_n ; 
	thrust::transform(d_val, d_val  + size, that.d_val, result.d_val, thrust::plus<int>());

	//D_Matrix result();
	return result;
}



//////////////////////////////////////////////////////////////////////////////////
// Exercice 2
bool D_Matrix::Exo2IsDone() {
	return true ;
}

class MatrixTransposeIteratorFunctor : public thrust::unary_function<int,int> {
	public:
		__host__ __device__ int operator()(const int &idx) {
			int row = idx/size ;
			int col = idx%size ;
			return col * size + row ;
		}

		MatrixTransposeIteratorFunctor(int size) : size(size){}

	private :
		int size ;

};

D_Matrix D_Matrix::transpose() const
{
	D_Matrix result(m_n);
	const int size = m_n * m_n ;

	// Transposing : 
	thrust::scatter(
		d_val,									// begin()
		d_val + size, 							// end()
		thrust::make_transform_iterator(		// iterator based on our functor
			thrust::make_counting_iterator(0),
			MatrixTransposeIteratorFunctor(m_n)
		),
		result.d_val							// return 
	) ;
	
	return result;
}



//////////////////////////////////////////////////////////////////////////////////
// Exercice 3
bool D_Matrix::Exo3IsDone() {
	return true ;
}

class DiffusionIteratorFunctor : public thrust::unary_function<int,int> {
	public:
		__device__ int operator()(const int &idx) {
			return d_val[ idx % size] ;
		}

		DiffusionIteratorFunctor(const thrust::device_ptr<int> val, int size) 
			: size(size), d_val(val)
			{}

	private :
		int size ;
		const thrust::device_ptr<int> d_val ;
	
};

void D_Matrix::diffusion(const int line, D_Matrix& result) const 
{

	thrust::copy_n( 
		thrust::make_transform_iterator(
			thrust::make_counting_iterator(0), 
			DiffusionIteratorFunctor(d_val + m_n * line, m_n)), 
		m_n * m_n, 
		result.d_val
	);
}

//////////////////////////////////////////////////////////////////////////////////
// Exercice 4
bool D_Matrix::Exo4IsDone() {
	return false;
}
// returns this times that ...
D_Matrix D_Matrix::product1(const D_Matrix& that) const
{	
/*
	D_Matrix result(m_n);
	
	D_Matrix C(m_n) ;
	D_Matrix D(m_n) ;
	D_Matrix TBi(m_n) ;
	D_Matrix TB = B.transpose();
	thrust::device_vector<int> colonne(m_n) ;
	thrust::device_vector<int> keys(m_n) ;
	
	for (int i = 0 ; i < m_n ; i++) {
		// D = product reasult of a for each column of B 
		TB.diffusion(i, TBi) ;

		transposedB.diffusion(i, tmp) ;
		thrust::transform(
			d_val, 
			d_val + m_n * m_n, 
			TBi.d_val
			D.d_val, 
			thrust::multiplies<int>()
		);
		
		auto fKeys = (thrust::placeholders::_1 / m_n) *m_n + i ;

		// Reduction with keys saved in a vector 
		thrust::device_vector<int> keys(m_n) ;
		D_Matrix segmentedReduced (m_n);
		thrust::fill(keys.begin(), keys.end(), i);
		thrust::reduce_by_key(
			thrust::make_transform_iterator(
				thrust::lake_counting_iterator(0) fKeys
			),
			thrust::device_vector
			segmentedReduced.d_val
		) ;

		// Scatterization to place results in good C columns 
		
		
	}

	return result;
*/
	D_Matrix bruh(m_n);
	return bruh ;
}

//////////////////////////////////////////////////////////////////////////////////
// Exercice 5
bool D_Matrix::Exo5IsDone() {
	return false;
}
// returns this times that ...
D_Matrix D_Matrix::product2(const D_Matrix& that) const {
	return D_Matrix(m_n);
}
