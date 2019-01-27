#include "student2.hpp"
#include "../utils/chronoGPU.hpp"
// KERNEL 1 : RGB TO HSV VALUES (in fact, just V values, and not even divided by 255)
__global__ void RGB2VKernel(uchar * input, int * hsv_v, int image_width, int image_height) {
    // Getting the local TID 
    int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid > image_width * image_height * 3) return ;

    // caching the Thread ID to have the G and B values
    int local_tid = threadIdx.x ;

    __shared__ uchar rgb_cache[576] ;
    if (local_tid < 576) rgb_cache[local_tid] = input[tid] ;
    __syncthreads() ;

    // verifying if the r thread is not in conflict with the g and b one : 
    if (local_tid % 3 != 0) return  ;

    // getting rgb values : 
    int r = (int) rgb_cache[local_tid];
    int g = (int) rgb_cache[local_tid + 1];
    int b = (int) rgb_cache[local_tid + 2];     

    // Choosing the highest one (V = CMax)
    int max_color = max(max(r, g), b) ; 

    // returning the V value
    hsv_v[int(tid / 3)] = max_color ;   
}

// KERNEL 2 : Finding Neighbors and getting medians 
__global__ void FindNeighborsMedian(int * input, uchar * output, int image_width, int image_height, int window_size) {

    // getting the TID 
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int tid = x + y * image_width;
    if (x >= image_width || y >= image_height || tid > image_width * image_height) return ;

    // Making an array to stock V values of neighbors
    int neighbors_v_values[225];
    int neighbors_v_values_iterator = 0 ;

    // finding neighbors V values 
    for (int i = -(window_size/2) ; i <= (window_size/2) ; i++) {
        for (int j = -(window_size/2) ; j <= (window_size/2) ; j++) {
        
            // get the distant id 
            int distant_x = x + i ;
            int distant_y = y + j ;
            int distant_tid = distant_x + distant_y * image_width ;

            // testing if the distant value is still on the image        
            if (
                (distant_tid < image_width * image_height) &&
                (distant_tid >= 0)
            )  {

                neighbors_v_values[neighbors_v_values_iterator] = input[distant_tid] ;
                neighbors_v_values_iterator++ ;
            }
        }
    }

    // Sorting the neighbor values with a bubble sort (in-place sorting algorithme)
    for(int i = 9-1 ; i > 1  ; i--) {
        for (int j = 0 ; j < i-1 ; j++) {
            if(neighbors_v_values[j+1] < neighbors_v_values[j]) {
                // replace j+1 by J
                int tmp = neighbors_v_values[j] ;
                neighbors_v_values[j] = neighbors_v_values[j+1] ;
                neighbors_v_values[j+1] = tmp ;
            }
        }
    }

    // returning the median value of the array 
    output[tid] = (uchar) neighbors_v_values[neighbors_v_values_iterator / 2] ;
    
}


// MAIN FUNCTION 
float student2(const PPMBitmap &in, PPMBitmap &out, const int size) { 

    // Usefull variables 
    int pixel_count = in.getWidth() * in.getHeight() ;
    int pointer_size = pixel_count * 3 ;

    int block_size  = 24 * 24 ; 
    int grid_size   = ceil(pointer_size / block_size) ;
    dim3 block_size2D = dim3(32, 32) ;
    dim3 grid_size2D = dim3(ceil(in.getWidth() / block_size2D.x), ceil(in.getHeight() / block_size2D.y));
    
    // Device arrays 
    uchar * device_input ;
    int * device_hsv_v ;
    uchar * device_output ; 

    // Host arrays 
    uchar * pixel_ptr   = in.getPtr() ;
    uchar * host_output = (uchar *)malloc(sizeof(uchar) * pixel_count) ;


    // Allocation of device arrays
    cudaMalloc(&device_input, sizeof(uchar) * pointer_size) ;
    cudaMalloc(&device_hsv_v, sizeof(int)   * pixel_count) ;
    cudaMalloc(&device_output, sizeof(uchar) * pixel_count) ;

    // Copying input values to device 
    cudaMemcpy(device_input, pixel_ptr, sizeof(uchar) * pointer_size, cudaMemcpyHostToDevice) ;
	
	ChronoGPU chGPU ; chGPU.start() ;

    // Getting the V values in device_hsv_v
    RGB2VKernel<<<grid_size, block_size>>>(device_input, device_hsv_v, in.getWidth(), in.getHeight());

    // Computing neighbors median v values
    FindNeighborsMedian<<<grid_size2D, block_size2D>>>(device_hsv_v, device_output, in.getWidth(), in.getHeight(), size) ;

	chGPU.stop() ;

    // Copying the results : 
    cudaMemcpy(host_output, device_output, sizeof(uchar) * pixel_count, cudaMemcpyDeviceToHost) ;


    // copying the results to the bitmap image 

    for (int i = 0 ; i < pixel_count ; i++){
        int x = i % in.getWidth() ;
        int y = i / in.getWidth() ; 
        uchar pixel_value = host_output[i] ;
        PPMBitmap::RGBcol rgb(pixel_value, pixel_value, pixel_value) ; 
        out.setPixel(x, y, rgb);  
    }


    // Freeing memory
    cudaFree(device_input) ;
    cudaFree(device_hsv_v) ;
    cudaFree(device_output) ;

    return chGPU.elapsedTime() ;
}
