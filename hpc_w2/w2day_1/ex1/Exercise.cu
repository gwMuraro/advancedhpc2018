#include "Exercise.hpp"
#include "include/chronoGPU.hpp"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>

// === Functor for question 1, can be replace by thrust::plus<int>()
class AdditionFunctor : public thrust::binary_function<int,int,int> {
	public:
		__host__ __device__ int operator()(const int &x, const int &y) {
			return x + y ;
		}

};

// === Functor for question 3
typedef thrust::tuple<int, int, int> Question3_tuple ;
class AdditionFunctor3 : public thrust::unary_function<Question3_tuple, int> {
	public : 
		__host__ __device__ int operator() (const Question3_tuple &t) {
		
			return thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<2>(t) ;
		}

};


void Exercise::Question1(const thrust::host_vector<int>& A,
						 const thrust::host_vector<int>& B, 
						 thrust::host_vector<int>&C) const
{

	ChronoGPU chrUP, chrDOWN, chrGPU;

	for (int i = 0 ; i < 3 ; i++) {
		chrUP.start();	
		thrust::device_vector<int> X(A);
		thrust::device_vector<int> Y(B);
		thrust::device_vector<int> Z(A.size());		
		chrUP.stop();

		chrGPU.start();
		thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), thrust::plus<int>());
		chrGPU.stop();
	
		chrDOWN.start();
		C = Z ;
		chrDOWN.stop();
	
	}

	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();

	std::cout << "Q1 elapsed in " 	<< elapsed 				<< "ms" << std::endl ;
	std::cout << "- UP time " 		<< chrUP.elapsedTime() 	<< "ms" << std::endl ;
	std::cout << "- GPU  " 			<< chrGPU.elapsedTime() << "ms" << std::endl ;
	std::cout << "- Down time " 	<< chrDOWN.elapsedTime() << "ms\n" << std::endl ;
		 
}


void Exercise::Question2(thrust::host_vector<int>&A) const {
	
	ChronoGPU chrUP, chrDOWN, chrGPU;
	
	for (int i = 3 ; i-- ; ) {
		// Declaration & allocation
		chrUP.start() ;
		thrust::device_vector<int> d_sequentials(A.size());
		thrust::device_vector<int> d_constants(A.size());
		thrust::device_vector<int> d_results(A.size());

		// filling 
		thrust::sequence(d_sequentials.begin(), d_sequentials.end(), 1.f) ;
		thrust::fill(d_constants.begin(), d_constants.end(), 4); // 2¹⁶ in theory
		chrUP.stop() ;	
	
		// Mapping 
		chrGPU.start();
		thrust::transform(d_sequentials.begin(), d_sequentials.end(), d_constants.begin(), d_results.begin(), thrust::plus<int>());
		chrGPU.stop() ;
	
		// Copying the results 
		chrDOWN.start() ;
		A = d_results ;
		chrDOWN.stop() ;
	}	
	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime() ;

	std::cout << "Q2 elapsed time : " << elapsed 		<< " ms." << std::endl ;
	std::cout << "- UP time   : " << chrUP.elapsedTime() 	<< " ms." << std::endl ;
	std::cout << "- GPU time  : " << chrGPU.elapsedTime() 	<< " ms." << std::endl ;
	std::cout << "- DOWN time : " << chrDOWN.elapsedTime() 	<< " ms.\n" << std::endl ;

	
}



void Exercise::Question3(const thrust::host_vector<int>& A,
						 const thrust::host_vector<int>& B, 
						 const thrust::host_vector<int>& C, 
						 thrust::host_vector<int>&D) const 
{

	ChronoGPU chrUP, chrDOWN, chrGPU;
	
	for (int i = 3 ; i-- ; ) {
		// Declaration & allocation
		chrUP.start() ;
		thrust::device_vector<int> d_A(A);
		thrust::device_vector<int> d_B(B);
		thrust::device_vector<int> d_C(C);
		thrust::device_vector<int> d_results(A.size());

		// filling 
		
		chrUP.stop() ;	
	
		// Mapping 
		chrGPU.start();
		int size = A.size();
		thrust::transform(  thrust::make_zip_iterator(thrust::make_tuple(d_A.begin(), d_B.begin(), d_C.begin())), 
							thrust::make_zip_iterator(thrust::make_tuple(d_A.end(), d_B.end(), d_C.end())), 
							d_results.begin(), 
							AdditionFunctor3());
		chrGPU.stop() ;
	
		// Copying the results 
		chrDOWN.start() ;
		D = d_results ;
		chrDOWN.stop() ;
	}	
	float elapsed = chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime() ;

	std::cout << "Q3 elapsed time : " << elapsed 		<< " ms." << std::endl ;
	std::cout << "- UP time   : " << chrUP.elapsedTime() 	<< " ms." << std::endl ;
	std::cout << "- GPU time  : " << chrGPU.elapsedTime() 	<< " ms." << std::endl ;
	std::cout << "- DOWN time : " << chrDOWN.elapsedTime() 	<< " ms.\n" << std::endl ;

}
