#include "student1.hpp"
#include "../utils/chronoGPU.hpp"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


//==============IMPLEMENTATION OF RGB2HSV FROM PREVIOUS LABWORK===============
// converts a RGB color to a HSV one ...
__device__
float3 RGB2HSV( const uchar3 inRGB ) {
	const float R = (float)( inRGB.x ) / 256.f;
	const float G = (float)( inRGB.y ) / 256.f;
	const float B = (float)( inRGB.z ) / 256.f;

	const float min		= fminf( R, fminf( G, B ) );
	const float max		= fmaxf( R, fmaxf( G, B ) );
	const float delta	= max - min;

	// H
	float H;
	if( delta < FLT_EPSILON )
		H = 0.f;
	else if	( max == R )
		H = 60.f * ( G - B ) / ( delta + FLT_EPSILON )+ 360.f;
	else if ( max == G )
		H = 60.f * ( B - R ) / ( delta + FLT_EPSILON ) + 120.f;
	else
		H = 60.f * ( R - G ) / ( delta + FLT_EPSILON ) + 240.f;
	while	( H >= 360.f )
		H -= 360.f ;

	// S
	float S = max < FLT_EPSILON ? 0.f : 1.f - min / max;

	// V
	float V = max;

	return make_float3(H, S, V);
}


// converts a HSV color to a RGB one ...
__device__
uchar3 HSV2RGB( const float H, const float S, const float V )
{
	const float	d	= H / 60.f;
	const int	hi	= (int)d % 6;
	const float f	= d - (float)hi;

	const float l   = V * ( 1.f - S );
	const float m	= V * ( 1.f - f * S );
	const float n	= V * ( 1.f - ( 1.f - f ) * S );

	float R, G, B;

	if		( hi == 0 )
		{ R = V; G = n;	B = l; }
	else if ( hi == 1 )
		{ R = m; G = V;	B = l; }
	else if ( hi == 2 )
		{ R = l; G = V;	B = n; }
	else if ( hi == 3 )
		{ R = l; G = m;	B = V; }
	else if ( hi == 4 )
		{ R = n; G = l;	B = V; }
	else
		{ R = V; G = l;	B = m; }

	return make_uchar3(R*256.f, G*256.f, B*256.f);
}

//================================================================

//===================FUNCTORS FOR EXERCISE #1===================== 
/**	FUNCTOR 1 :
*	From the ID (int) (counting iterator), we get the rgb values in an array 
*	of uchar gave in parameter. 
*	With this RGB value we compute the HSV convertion with a given 
*	RGB2HSV function.
* 	We finally return the V value of HSV (float).
*	
*	@param idx : the id of the targeted pixel of the image
* 	@return hsv_value.z : the V value of HSV 
*/
class ConvertToHSVFunctor : public thrust::unary_function<int, float>{
	public : 
		__device__ float operator()(const int &idx){
			// get the pixel id in the pixels array
			int pixel_tid = 3 * idx ; 
			
			// get RGB values 
			uchar3 rgb_values = make_uchar3(pixels[pixel_tid], pixels[pixel_tid + 1], pixels[pixel_tid + 2]);
			
			// get the conerted HSV value 
			float3 hsv_values = RGB2HSV(rgb_values) ;
			
			// returning only the value 
			return hsv_values.z;
		}
		
		ConvertToHSVFunctor(uchar * pixels) : pixels(pixels) {}
		
	private : 
		uchar * pixels ; // device_vector casted in pointer 
		
};

/**
*	This big boy Compute the median filter by finding the neighbors of the pixel "idx"
*	in the device_vector passed in parameter
* 	@return median_v_value : the median value between the treated pixel and his neighbors
*/
class ComputeMedianFilterFunctor : public thrust::unary_function<int, float>{
	public : 
		__device__ float operator() (const int &idx) {
			// Getting the local coordinates 
			int local_tid_x = idx % image_width ;
			int local_tid_y = idx / image_width ;
		
			
			// Making the array for neighbors' V values 
			float neighbors_v_value[225] ;
			int   neighbors_v_value_iterator = 0 ;
			
			// Finding the neighbors (O(n^2) = 9 iterations)
			for (int i = -(window_size/2) ; i <= (window_size/2) ; i++){
				for (int j = -(window_size/2) ; j <=(window_size/2) ; j++){

					// Geting the neighbor coordinates 
					int distant_tid_x = local_tid_x + i ;
					int distant_tid_y = local_tid_y + j ;
					int distant_tid   = distant_tid_x + distant_tid_y * image_width ;
					
					// testing if the distant pixel is still in the image
					if (
						distant_tid_x >= 0 && 
						distant_tid_x < image_width && 
						distant_tid_y >= 0 && 
						distant_tid_y < image_height
					) {
						// Getting the V value of our distant pixel 
						neighbors_v_value[neighbors_v_value_iterator] = device_v_values[distant_tid] ;
						neighbors_v_value_iterator++ ; 
					}
				}
			}

			// Making a bubble sort (complexity O(n^2) at worst -> 81 iterations)
			for (int i = (window_size*window_size) -1 ; i > 1 ; i--) {
				for (int j = 0 ; j < i - 1 ; j++) {
					if (neighbors_v_value[j+1] < neighbors_v_value[j]) {
						// replacing j+1 by j
						float tmp = neighbors_v_value[j] ; 
						neighbors_v_value[j] = neighbors_v_value[j+1] ;
						neighbors_v_value[j+1] = tmp ;
					}
				}
			}
			return neighbors_v_value[neighbors_v_value_iterator / 2] ;
		}
		
		ComputeMedianFilterFunctor(float * device_v_values, int image_width, int image_height, int window_size) : 
			device_v_values(device_v_values),
			image_width(image_width), 
			image_height(image_height),
			window_size(window_size) {}
		
	private : 
		int image_width ;
		int image_height ;
		int window_size ;
		float * device_v_values ;
};		

class ConvertToRGBFunctor : public thrust::binary_function<int, float, uchar3> {
	public : 
		__device__ uchar3 operator() (const int &idx, const float &v_value) {
			// get local id in pixels 
			int pixel_id = idx * 3 ; 
			
			// convert to HSV 
			uchar3 old_rgb_pixel = make_uchar3(pixels[pixel_id], pixels[pixel_id + 1], pixels[pixel_id + 2]) ;
			float3 old_hsv_pixel = RGB2HSV(old_rgb_pixel) ;
		
			// convert to RGB and changing the V value 
			uchar3 new_rgb_pixel = HSV2RGB(old_hsv_pixel.x, old_hsv_pixel.y, v_value) ;
			
			// set the new value to the device_vector "pixels"
			pixels[pixel_id] = new_rgb_pixel.x ;
			pixels[pixel_id + 1] = new_rgb_pixel.y ;
			pixels[pixel_id + 2] = new_rgb_pixel.z ;

			return new_rgb_pixel ;
		}
		
		ConvertToRGBFunctor(uchar * pixels) : pixels(pixels) {}
		
	private : 
		uchar * pixels ; 
} ;

/* Exercice 1.
* Here, you have to apply the Median Filter on the input image.
* Calculations have to be done using Thrust on device. 
* You have to measure the computation times, and to return the number of ms 
* your calculation took (do not include the memcpy).
*
* @param in: input image
* @param out: output (filtered) image
* @param size: width of the kernel 
*/
float student1(const PPMBitmap &in, PPMBitmap &out, const int size) {

	// get the pointer of our pixels RGB values 
	uchar * pixels = in.getPtr();
	
	// getting the sizes of our objects 
	int ptr_size = in.getWidth() * in.getHeight() * 3 ; 
	int image_size = in.getWidth() * in.getHeight() ;
	
	// making some vectors 
	thrust::host_vector<uchar>   host_in (pixels, pixels + ptr_size); 
	thrust::device_vector<uchar> device_in (host_in);
	thrust::device_vector<float> device_hsv_values (image_size) ;
	thrust::device_vector<float> device_hsv_arranged_values (image_size) ;	
	thrust::device_vector<uchar3> device_rgb_out(image_size) ;
	thrust::host_vector<uchar3>   host_out (image_size); 
	

	ChronoGPU chGPU ; chGPU.start(); 	// start Chrono
	
	// Making an iterator to process 3 by 3 values to convert RGB to V values
	thrust::transform(
		thrust::make_counting_iterator(0), 
		thrust::make_counting_iterator(image_size), 
		device_hsv_values.begin(), 
		ConvertToHSVFunctor(thrust::raw_pointer_cast(device_in.data()))
	) ;
	
	// From V values, compute the neighbors
	thrust::transform(
		thrust::make_counting_iterator(0), 
		thrust::make_counting_iterator(image_size), 
		device_hsv_arranged_values.begin(), 
		ComputeMedianFilterFunctor(
			thrust::raw_pointer_cast(device_hsv_values.data()), 
			in.getWidth(), 
			in.getHeight(), 
			size
		) 
	) ;
	
	// Convert HSV arranged to RGB by passing the device_vector pointer of pixels 
	thrust::transform(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(image_size), 
		device_hsv_arranged_values.begin(), 
		device_rgb_out.begin(), 
		ConvertToRGBFunctor(thrust::raw_pointer_cast(device_in.data()))
	) ;
	
	chGPU.stop() ;	
		
	// repixeling the initial image 
	thrust::copy(device_rgb_out.begin(), device_rgb_out.end(), host_out.begin()) ;
	
	// we could have mapped it without structures
	for (int i = 0 ; i < image_size ; i++) {

		int tid_x = i % in.getWidth() ;
		int tid_y = i / in.getWidth() ; 
		PPMBitmap::RGBcol that_thing_that_avoid_us_to_use_uchar3 = PPMBitmap::RGBcol(host_out[i].x, host_out[i].y, host_out[i].z);

		out.setPixel(tid_x, tid_y, that_thing_that_avoid_us_to_use_uchar3);
	}
	
	
	return chGPU.elapsedTime() ;
}
