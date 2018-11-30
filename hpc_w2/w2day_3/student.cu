#include <iostream>
#include "student.hpp"

// do not forget to add the needed included files
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "utils/chronoGPU.hpp"


// ==========================================================================================
// Exercise 1

class ReturnBlueObjects : public thrust::unary_function<ColoredObject ,int> {
	public:
		__host__ __device__ int operator()(ColoredObject c) {
			return c.color == ColoredObject::BLUE  ;
		}

};

// mandatory function: should returns the blue objects contained in the input parameter
thrust::host_vector<ColoredObject> compactBlue( const thrust::host_vector<ColoredObject>& input ) {
	
	ChronoGPU chrUP, chrDOWN, chrGPU ;

	chrUP.start();
	thrust::device_vector<ColoredObject> d_output(input) ;	
	thrust::device_vector<ColoredObject> d_input(input) ;
	chrUP.stop();
	
	chrGPU.start(); 
	// map traduction of the question "is my element i is blue ?"
	thrust::device_vector<int> predicates (input.size());
	thrust::transform(d_input.begin(), d_input.end(), predicates.begin(), ReturnBlueObjects());
	
	// Prefix-sum (or scan) the predicates to have the index for the scatter to put at right place  
	thrust::device_vector<int> indexs (predicates.size());
	thrust::exclusive_scan(predicates.begin(), predicates.end(), indexs.begin());
	
	// Réduire et sélectionner avec un scatter_if 
	thrust::scatter_if( d_input.begin(), d_input.end(), indexs.begin(), predicates.begin(), d_output.begin() );
	chrGPU.stop() ;
	
	chrDOWN.start();
	// utilser un copy_n 
	int last_out_value = indexs[indexs.size() -1] ; 					// getting the last usefull value index
	thrust::host_vector<ColoredObject> answer( last_out_value );		// allocating host vector
	thrust::copy_n( d_output.begin(), last_out_value, answer.begin() );	// copying device to host 
	chrDOWN.stop();
		
		
	std::cout << "Question 1 elapsed in \t"	<< chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime() << " ms." << std::endl ;
	std::cout << "\t- chrUP : \t" 			<< chrUP.elapsedTime()<< " ms." << std::endl ;	
	std::cout << "\t- chrGPU : \t" 			<< chrGPU.elapsedTime()<< " ms." << std::endl ;	
	std::cout << "\t- chrDOWN : \t" 		<< chrDOWN.elapsedTime()<< " ms." << std::endl ;	
	return answer;
}

// ==========================================================================================
// Exercise 2

class BitIsAtOneFunctor : public thrust::unary_function<int, int> {
	public:
		__host__ __device__ int operator()(int number) {
			return number >>(decalage) & 0x01    ;
		}
		
		BitIsAtOneFunctor(int decalage) : decalage(decalage){}

	private : 
		int decalage ;

};

class ReverseBitsFunctor : public thrust::unary_function<int, int> {
	public:
		__host__ __device__ int operator()(int number) {
			return number == 0 ; // we do the inversion here 
		}
};
/*
class ScatterTransformFunctor : public thrust::binary_function<int, int, int>{
	public : 
		__host__ __device__ int operator() (int number, int flag) {
			return flag == 1   ;
		}
};
*/

thrust::host_vector<int> radixSort( const thrust::host_vector<int>& input ) {

	
	ChronoGPU chrUP, chrDOWN, chrGPU, chrCHEAT ; 
	
	chrUP.start() ;
	thrust::host_vector<int> answer;
	thrust::device_vector<int> d_input(input); 
	thrust::device_vector<int> d_input2(input); 
	thrust::device_vector<int> flags(input.size());
	thrust::device_vector<int> flags_reversed(input.size());	
	thrust::device_vector<int> index_down(input.size());
	thrust::device_vector<int> index_up(input.size()) ;
	thrust::device_vector<int> adding_to_iup(input.size());
	thrust::device_vector<int> d_output(input.size()) ;
	chrUP.stop() ;	
	
	chrCHEAT.start();
	thrust::sort(d_input2.begin(), d_input2.end()) ; // to cheat
	chrCHEAT.stop() ;
	
	chrGPU.start();
	for (int i = 0 ; i < 32 ; i++){
		// Map to obtain if the bit is on 1 or 0 
		thrust::transform(d_input.begin(), d_input.end(), flags.begin(), BitIsAtOneFunctor(i));
		thrust::transform(flags.begin(), flags.end(), flags_reversed.begin(), ReverseBitsFunctor());
		
		// reverse exclusive scan to make the 0s on first cols -> define idown 
		thrust::exclusive_scan(flags_reversed.begin(), flags_reversed.end(), index_down.begin());
		
		// define iup : 
		thrust::exclusive_scan(flags.begin(), flags.end(), index_up.begin());
		thrust::fill(adding_to_iup.begin(), adding_to_iup.end() - 1, index_down[index_down.size()-1] + 1);
		thrust::transform(index_up.begin(), index_up.end(), adding_to_iup.begin(), index_up.begin(), thrust::plus<int>());

		// Scatter if 
		// sort the first part 
		thrust::scatter_if(d_input.begin(), d_input.end(), index_down.begin(), flags_reversed.begin(), d_output.begin());
		// sort the secondpart
		thrust::scatter_if(d_input.begin(), d_input.end(), index_up.begin(), flags.begin(), d_output.begin());
		
	
		
		if (i == 31){
	//		for (int j = 1048400; j< 1048400 + 32 ; j++) { 
	//			std::cout << "Bruh "<< index_down[index_down.size() - 1] <<" : " << d_output[j] << std::endl ;
	//		}
			std::cout << "Bruh "<< index_down[index_down.size() - 1] <<" : " << index_up[0] << std::endl ;
		}
		d_input = d_output ;
	}

	chrGPU.stop();
	
	chrDOWN.start() ;
	answer = d_output;
	chrDOWN.stop() ;	
	/*
	for (int i = 0 ; i < 1048576 ; i ++) {
		if (answer[i] != d_input2[i] && i == 1048575){
			std::cout << "error at index : " << i << "[" << answer[i] <<"!=" << d_input2[i] << std::endl ; 
		//	break ;
		}
	}
	*/
	std::cout << "Question 2 elapsed in \t"	<< chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime() << " ms." << std::endl ;
	std::cout << "\t- chrUP : \t" 			<< chrUP.elapsedTime()<< " ms." << std::endl ;	
	std::cout << "\t- chrGPU : \t" 			<< chrGPU.elapsedTime()<< " ms." << std::endl ;	
	std::cout << "\t- chrDOWN : \t" 		<< chrDOWN.elapsedTime()<< " ms." << std::endl ;	
	std::cout << "\t- chrCHEAT : \t" 		<< chrCHEAT.elapsedTime()<< " ms." << std::endl ;	
	
	return answer;
}

// ==========================================================================================
// Exercise 3
// Feel free to add any function you need, it is your file ;-)
thrust::host_vector<int> quickSort( const thrust::host_vector<int>& h_input ) {
	thrust::host_vector<int> answer;
	return answer;
}
