#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include "Exercise.hpp"
#include "include/chronoGPU.hpp"

// Question 1 : functor for gather 
class EvenIteratorFunctor : public thrust::unary_function<int,int> {
	public:
		__host__ __device__ int operator()(const int &x) {
			return (x < max_size/2)? 2*x : 2*x - max_size + 1   ;
		}
		
		
		EvenIteratorFunctor(int size) : max_size(size) {} // constructor with initializer
	
	private :
		int max_size;
};

// Question 2 : functor for Scatter
class EvenIteratorScatterFunctor : public thrust::unary_function<int,int> {
	public:
		__host__ __device__ int operator()(const int &x) {
			return (x%2 == 0) ? x/2 : (max_size/2 + x/2) ;
		}
		
		EvenIteratorScatterFunctor(int size) : max_size(size) {} // constructor with initializer
	
	private :
		int max_size;
};

void Exercise::Question1(const thrust::host_vector<int>& A,
						 thrust::host_vector<int>& OE ) const
{


	// Use transform iterator -> index from where you want to write
	// odd formula : 1 + 2i - n
	// use counting iterator to knw what index is processing OR class variable (we use Functor-class variable)

	ChronoGPU cUP, cIT, cDOWN, cGPU;	
	
	for (int i= 3 ; i-- ; ){

		// Making a device vector 
		cUP.start() ;
		thrust::device_vector<int> d_A (A);
		thrust::device_vector<int> d_OE (A.size());
		cUP.stop() ;	

		// Making an iterator with even values until A.size()/2 and odd for others. 
		cIT.start();
		typedef thrust::device_vector<int>::iterator IntIterator; 
		thrust::transform_iterator<EvenIteratorFunctor, IntIterator> d_OE_iterator(d_A.begin(), EvenIteratorFunctor(A.size()));
		cIT.stop();	

		//Gathering
		cGPU.start();
		thrust::gather(d_OE_iterator, d_OE_iterator + A.size(), d_A.begin(), d_OE.begin() );
		cGPU.stop();

		// copying results 
		cDOWN.start();
		OE = d_OE;
		cDOWN.stop() ;
	}
	// Displaying
	float elapsed = cUP.elapsedTime() + cIT.elapsedTime() + cDOWN.elapsedTime() + cGPU.elapsedTime();
	std::cout << "Question 1 ellapsed in " 	<< elapsed 		<< " ms" << std::endl ;
	std::cout << "\t- UP Time : " 		<< cUP.elapsedTime() 	<< " ms" << std::endl ;
	std::cout << "\t- Iteratoriz° time : " 	<< cIT.elapsedTime() 	<< " ms" << std::endl ;
	std::cout << "\t- GPU time : " 		<< cGPU.elapsedTime()	<< " ms" << std::endl ;
	std::cout << "\t- DOWN time : " 	<< cDOWN.elapsedTime()	<< " ms\n" << std::endl ;
}



void Exercise::Question2(const thrust::host_vector<int>&A, 
						thrust::host_vector<int>&OE) const 
{
	ChronoGPU cUP, cIT, cDOWN, cGPU;	
	
	for (int i= 1 ; i-- ; ){

		// Making a device vector 
		cUP.start() ;
		thrust::device_vector<int> d_A (A);
		thrust::device_vector<int> d_OE (A.size());
		cUP.stop() ;	

		// Making an iterator with even values until A.size()/2 and odd for others. 
		cIT.start();
		typedef thrust::device_vector<int>::iterator IntIterator; 
		thrust::transform_iterator<EvenIteratorScatterFunctor, IntIterator> d_OE_iterator(d_A.begin(), EvenIteratorScatterFunctor(A.size()));
		cIT.stop();	

		//Gathering
		cGPU.start();
		thrust::scatter(thrust::device, d_A.begin(), d_A.end(), d_OE_iterator, d_OE.begin() );
		cGPU.stop();

		// copying results 
		cDOWN.start();
		OE = d_OE;
		cDOWN.stop() ;		
	}
	// Displaying

	float elapsed = cUP.elapsedTime() + cIT.elapsedTime() + cDOWN.elapsedTime() + cGPU.elapsedTime();
	std::cout << "Question 2 ellapsed in " 	<< elapsed 		<< " ms" << std::endl ;
	std::cout << "\t- UP Time : " 			<< cUP.elapsedTime() 	<< " ms" << std::endl ;
	std::cout << "\t- Iteratoriz° time : " 	<< cIT.elapsedTime() 	<< " ms" << std::endl ;
	std::cout << "\t- GPU time : " 			<< cGPU.elapsedTime()	<< " ms" << std::endl ;
	std::cout << "\t- DOWN time : " 		<< cDOWN.elapsedTime()	<< " ms\n" << std::endl ;
	
}




template <typename T>
void Exercise::Question3(const thrust::host_vector<T>& A,
						thrust::host_vector<T>&OE) const 
{
  // TODO: idem for big objects
}


struct MyDataType {
	MyDataType(int i) : m_i(i) {}
	MyDataType() = default;
	~MyDataType() = default;
	int m_i;
	operator int() const { return m_i; }

	// TODO: add what you want ...
};

// Warning: do not modify the following function ...
void Exercise::checkQuestion3() const {
	const size_t size = sizeof(MyDataType)*m_size;
	std::cout<<"Check exercice 3 with arrays of size "<<(size>>20)<<" Mb"<<std::endl;
	checkQuestion3withDataType(MyDataType(0));
}
