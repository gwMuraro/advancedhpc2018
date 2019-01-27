#include "student3.hpp"
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
/*
* You have here to compute the segmented image from the filtered one.
* Calculations have to be done on Device using Thrust.
*
* @param in: input (filtered) image
* @param out: output (segmented) image
* @param threshold: thresholding value (remove the edges greater than it)
*/
/*	FUNCTOR 1 : 
*	Convert to gray scale the pixels (3 uchars values). 
*	The gray scale are given in int in a array 
*	whose size is a third of the initial size.
*/
class ConvertInGrayScaleFunctor : public thrust::unary_function<int,int> {

	public : 
		__device__ int operator()(const int &pixel_id) {

			// Getting the pointer position with the pixel_id
			int pixels_pointer_id = pixel_id * 3 ; 

			// Getting the rgb values 
			int red   = (int)pixels_pointer[pixels_pointer_id] ;
			int green = (int)pixels_pointer[pixels_pointer_id + 1] ;
			int blue  = (int)pixels_pointer[pixels_pointer_id + 2] ;

			// Making the aerage of values and returning them
			return (red + green + blue) / 3 ;
		}

		// Constructor with the uchar array 
		ConvertInGrayScaleFunctor(uchar * pixels_pointer) : 
			pixels_pointer(pixels_pointer) {}
	
	private : 
		uchar * pixels_pointer ; 
} ;


/*	FUNCTOR 2 : 
*	With that functor, we get the pixel id and return it adjascent neighbors,
*	according to their weight that do not overpass the threshold
*	The adjascent neighbors are stored in an uchar4 (max. 4 neighbors). 
*	In fine, the uchar4 is on the pixel's id position on the returned device_vector. 
*/
class GetConnectedAdjascentNeighbors : public thrust::unary_function<int, int4>{

	public :
		__device__ int4 operator()(const int &pixel_id) {

			// getting our local pixel x and y 
			int local_x = pixel_id % image_width ;
			int local_y = pixel_id / image_width ; 

			// we want to have the pixels with this coordinates 
			int distant_xs[] = {-1, 1, 0, 0} ;
			int distant_ys[] = {0, 0, -1, 1} ;
			int neighbors[] = {-1, -1, -1, -1} ;

			
			for (int i = 0 ; i < 4 ; i++){
				int distant_x = local_x + distant_xs[i] ;
				int distant_y = local_y + distant_ys[i] ;
				int distant_id = distant_x + distant_y * image_width ;

				// testing if they are into the image
				if (	distant_x >= 0			&&
						distant_x < image_width && 
						distant_y >= 0			&& 
						distant_y <image_height
				) {
	
					// computing the absolute difference 
					int diff = pixel_values[pixel_id] - pixel_values[distant_id];
					int abs_diff = diff < 0 ? -1 * diff : diff ;

					// testing if the difference of weights of the pixels is under the threshold
					if (abs_diff < threshold) {
						neighbors[i] = distant_id ;
					}
				}
			}

			// uchar composition : West, East, North, South
			// returning the uchar4 formatted adjascent neighbors
			return make_int4(neighbors[0], neighbors[1], neighbors[2], neighbors[3]) ;
		}

		// Constructor with the uchar array 
		GetConnectedAdjascentNeighbors(int * pixel_values, int threshold, int image_width, int pixel_count, int image_height) : 
			pixel_values(pixel_values), 
			threshold(threshold), 
			image_width(image_width),
			pixel_count(pixel_count), 
			image_height(image_height) {}
	
	private : 
		int * pixel_values ; 
		int threshold ;
		int image_width ;
		int pixel_count ;
		int image_height ;
} ;

/**	Device function to make a simili-tree by a recursive algorithm.
*	We try to make the tree by knowing the less-weighted-linked pixel to our local_pixel.
* 	then we try to find the minimal pixel_id that link all the previous ones to our local_pixel. 
* 	It returns finally the minimal "tree-id" based on the minimal pixel id found. 
*/
__device__ int findInitialLinkedPixel(int4 * neighbors, int * pixel_values, int local_id) {

	// Knowing the linked neighbors of the pixel
	int4 pixel = neighbors[local_id] ; 
	
	// Finding the linked pixels that are not -1 and adding it to an array
	int neighbor_pixel_ids[] = {-1, -1, -1, -1} ;
	int neighbor_pixel_ids_iterator = 0 ;
	
	if (pixel.x != -1 && pixel.x < local_id){
		neighbor_pixel_ids[neighbor_pixel_ids_iterator] = pixel.x ;
		neighbor_pixel_ids_iterator++ ;
	}
	if (pixel.y != -1 && pixel.y < local_id) {
		neighbor_pixel_ids[neighbor_pixel_ids_iterator] = pixel.y ;
		neighbor_pixel_ids_iterator++ ;	
	}
	if (pixel.z != -1 && pixel.z < local_id){
		neighbor_pixel_ids[neighbor_pixel_ids_iterator] = pixel.z ;
		neighbor_pixel_ids_iterator++ ;	
	}
	if (pixel.w != -1 && pixel.w < local_id){
		neighbor_pixel_ids[neighbor_pixel_ids_iterator] = pixel.w ;
		neighbor_pixel_ids_iterator++ ;	
	}
	
	if (neighbor_pixel_ids_iterator == 0) {
		return local_id ;
	} else {
		// choose the minimal linked pixel by his wieght
		int minimal_pixel_id = neighbor_pixel_ids[0] ;
		for (int i = 0 ; i < neighbor_pixel_ids_iterator ; i++){
			// if the weight of our current minimal pixel id is greater than the other,
			if (pixel_values[minimal_pixel_id] > pixel_values[neighbor_pixel_ids[i]]) {
				// the new minimal id becomes the id of the other one. 
				minimal_pixel_id = neighbor_pixel_ids[i];
			}
		}
		// We go on recursive 
		return findInitialLinkedPixel(neighbors, pixel_values, minimal_pixel_id) ;
	}	
}

/**	FUNCTOR 3 : making a tree 
*	convert pixels id to tree ids
*/
class BuildForestFunctor : public thrust::unary_function<int, int>{
	public : 
		__device__ int operator() (const int &idx) {
			return findInitialLinkedPixel(neighbors, gray_scale_pixels, idx) ;
		}
		BuildForestFunctor(int4 * neighbors, int * gray_scale_pixels) :
			neighbors(neighbors),
			gray_scale_pixels(gray_scale_pixels) {}
	private : 
		int4 * neighbors ; 
		int * gray_scale_pixels ;
} ;

/**	FUNCTOR 4 : Coloration 
*
*/
class ColorationFunctor : public thrust::unary_function<int, int>{
        public :
			__device__ int operator() (const int &origin_pixel) {
				return  pixels[origin_pixel];
			}

		ColorationFunctor(int * pixels) :
                        pixels(pixels) {}
        private :
                int * pixels ;
} ;



float student3(const PPMBitmap& in, PPMBitmap& out, const int threshold) {

	// Usefull variables 
	int pixel_count = in.getWidth() * in.getHeight() ;
	int pixel_pointer_size = pixel_count * 3 ; 
	uchar * pixel_pointer = in.getPtr() ; 
	
	// thrust vectors 
	thrust::host_vector<uchar> host_in(pixel_pointer, pixel_pointer + pixel_pointer_size) ;
	thrust::device_vector<uchar> device_in(host_in) ;
	thrust::device_vector<int> device_gray_scale(pixel_count) ;
	thrust::device_vector<int4> device_threshold_segmented(pixel_count) ;
	thrust::device_vector<int>  device_tree_ids(pixel_count) ;
	thrust::device_vector<int> device_out(pixel_count) ;
	thrust::host_vector<int> host_out(pixel_count);

	// transforms 
	ChronoGPU chGPU ; chGPU.start() ;

	// Functor 1 : ConvertRGB to Grayscale values in order to simplify the manipulations	
	thrust::transform(
		thrust::make_counting_iterator(0), 
		thrust::make_counting_iterator(pixel_count),
		device_gray_scale.begin(), 
		ConvertInGrayScaleFunctor(thrust::raw_pointer_cast(device_in.data()))
	) ;

	// Functor 2 : Finding the neighbors that are "connected" (threshold segmentation)
	thrust::transform(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(pixel_count),
		device_threshold_segmented.begin(), 
		GetConnectedAdjascentNeighbors(
			thrust::raw_pointer_cast(device_gray_scale.data()),
			threshold,
			in.getWidth(),
			pixel_count, 
			in.getHeight()
		)
	) ;

	// Functor 3 : Making a "forest" by associating each pixel a tree id 
	thrust::transform(
		thrust::make_counting_iterator(0), 
		thrust::make_counting_iterator(pixel_count),
		device_tree_ids.begin(),
		BuildForestFunctor(
			thrust::raw_pointer_cast(device_threshold_segmented.data()),
			thrust::raw_pointer_cast(device_gray_scale.data())	
		)
	) ;

	// Functor 4 : The pixel at the origin will give its color to the others 
	thrust::transform(
		device_tree_ids.begin(), 
		device_tree_ids.end(), 
		device_out.begin(),
		ColorationFunctor(thrust::raw_pointer_cast(device_gray_scale.data()))
	) ;

	chGPU.stop() ;
	
	// Copying the output to host
	thrust::copy(device_out.begin(), device_out.end(), host_out.begin()); 

	for (int i = 0 ; i < pixel_count ; i++) {
		int x = i % in.getWidth() ; 
		int y = i / in.getWidth() ; 
		out.setPixel(
			x,
			y,
			PPMBitmap::RGBcol(host_out[i], host_out[i], host_out[i])
		) ;
	}

	return chGPU.elapsedTime();
}
