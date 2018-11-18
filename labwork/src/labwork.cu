#include <stdio.h>
#include <stdlib.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <include/jpegloader.h>
#include <math.h>

#define ACTIVE_THREADS 4

JpegInfo *inputImage2 ;


int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage> <block number || blocksize> <Blur intensity || brightness>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

	
     // GM : added argument for lab 3
	int blockSize = (argv[3] != NULL ? atoi(argv[3]) : 1024) ;
	int blockNumber = (argv[3] != NULL && atoi(argv[3]) < 33) ? atoi(argv[3]) : 32 ;
	int blurDim = (argv[3] != NULL ) ? atoi(argv[3]) : 3 ; 
	int brightness = (argv[4] != NULL && argv[4] != 0) ? atoi(argv[4]) : 100;
	int kuwaharaCoef = (argv[3] != NULL) ? atoi(argv[3]) : 15 ;

	float blendCoeficient ;
	
	// getting the new image 
	if (lwNum == 6) {
		if (argc == 6){
	
			char * inputFileName2 = argv[4];
			blendCoeficient = atof(argv[5]) ;

			JpegLoader jpegLoader ;
			inputImage2 = jpegLoader.load(inputFileName2);
		
			// Parameters check 
			if (! inputImage2 ) {
				printf("ERROR : file didn't load. Program stops.");
				return 0 ;
			}
			if (blendCoeficient && (blendCoeficient > 1 || blendCoeficient < 0)) {
				printf("ERROR : blendCoeficient > 1 || < 0. Program stops.") ;
				return 0 ;
			}
		} else {
		    printf("Usage: labwork 6 <inputImage> <blockNumber> <inputImage2> <blendCoeficient = [0.0, ..., 1]>\n");
		    return 0 ;
		}
	}


    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
	    labwork.labwork1_OpenMP_doublePragma();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            printf("labwork with double pragma ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
	    timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU(blockSize);
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU(blockNumber);
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            printf("labwork %d ellapsed %.1fms (CPU)\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-cpu-out.jpg");

	    // with global memory
            timer.start() ;
            labwork.labwork5_GPU_not_shared(blockNumber, blurDim) ;
            printf("labwork %d ellapsed %.1fms (GPU global memory)\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-gpu-out-not-shared.jpg");
            
            // with shared memory 
            timer.start() ;
            labwork.labwork5_GPU(blockNumber, blurDim);
	    printf("labwork %d ellapsed %.1fms (GPU shared memory)\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-gpu-out.jpg");            
            break;
            
        case 6:
	    timer.start() ; 
            labwork.labwork6_GPU_binarization(blockNumber);
            printf("labwork %da ellapsed %.1fms (binarization)\n", lwNum, timer.getElapsedTimeInMilliSec());
	    labwork.saveOutputImage("labwork6a-gpu-out.jpg");
			
	    timer.start() ;
            labwork.labwork6_GPU_brightness(blockNumber, brightness);
            printf("labwork %db ellapsed %.1fms (brightness control)\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork6b-gpu-out.jpg");
            
	    if (inputImage2) {
	        timer.start() ;
		labwork.labwork6_GPU_blending(blockNumber, inputImage2, blendCoeficient);
		printf("labwork %dc ellapsed %.1fms (blending)\n", lwNum, timer.getElapsedTimeInMilliSec());
		labwork.saveOutputImage("labwork6c-gpu-out.jpg");
	    }
            break;
            
        case 7:
	    timer.start() ;
            labwork.labwork7_GPU(32);
	    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());            
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
      
        	timer.start();
		labwork.labwork8_GPU_maths();
		//printf("labwork %d with maths ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());            
            
        	timer.start();
		labwork.labwork8_GPU_scatter(blockNumber);
		printf("labwork %d with scatter ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());            
			

		labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
        	timer.start();
            labwork.labwork9_GPU();
			printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());            
            labwork.saveOutputImage("labwork9-gpu-out.jpg");

            break;
        case 10:
		timer.start();
            labwork.labwork10_GPU(kuwaharaCoef);
			printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());            
            labwork.saveOutputImage("labwork10-gpu-out.jpg");

            break;
    }
    //printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma omp target teams num_teams(4)
    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {             // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }

}

void Labwork::labwork1_OpenMP_doublePragma() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {             // let's do it 100 times, otherwise it's too fast!
        #pragma omp parallel for
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }

}


int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
	int number_of_devices; 
	cudaGetDeviceCount(&number_of_devices);
	printf("Number of devices : %.1d\n", number_of_devices);
	for (int i = 0 ; i < number_of_devices ; i++) {
		cudaDeviceProp properties ;
		cudaGetDeviceProperties(&properties, i) ;

		printf("Device n°%d\n", i);
		printf("Name : %s\n", properties.name);
		printf("\tCore info : Clock rate : %d | ", properties.clockRate);
		printf("Number of cores : %d | ", getSPcores(properties));
		printf("Number of multiprocessors : %d | ", properties.multiProcessorCount);
		printf("Warp size : %d\n", properties.warpSize);
		
		printf("\tMemory info : \tClock rate : %d | ", properties.memoryClockRate);
		printf("Bus width : %d | ", properties.memoryBusWidth);
		printf("Band width : %dMB/s\n", 2*properties.memoryClockRate * (properties.memoryBusWidth / 8) / 1000000);
	}
}

// Kernel for labwork 3 
__global__ void grayScale(uchar3 *input, uchar3 *output) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
	output[tid].z = output[tid].y = output[tid].x;
}


void Labwork::labwork3_GPU(int blockSize) {

	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;
	// int blockSize = atoi(argv[3]); // replace by the parameter
	int numBlock = pixelCount/blockSize ;

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output)
	uchar3 * devInput ; 
	uchar3 * devGray ;
	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel 
	grayScale<<<numBlock, blockSize>>>(devInput, devGray) ; 
	
	// Gettting the results from GPU to CPU 
	cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);
	cudaFree(devGray);
	
}

// Kernel for labwork 4
__global__ void grayScale2D(uchar3 *input, uchar3 *output, int totalWidth) {
	
	//getting the pixel with the second dimension
	int tid = (threadIdx.x + blockIdx.x * blockDim.x) + (totalWidth * (threadIdx.y + blockIdx.y * blockDim.y));
	output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
	output[tid].z = output[tid].y = output[tid].x;
}


void Labwork::labwork4_GPU(int blockNumber) {
	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;

	// We set grid size and block size as dim3 variables
	dim3 gridSize = dim3(inputImage->width / blockNumber, inputImage->height / blockNumber);
	dim3 blockSize2 = dim3(blockNumber, blockNumber);
	

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output)
	uchar3 * devInput ; 
	uchar3 * devGray ;
	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	grayScale2D<<<gridSize, blockSize2>>>(devInput, devGray, inputImage->width) ; 
	
	// Gettting the results from GPU to CPU 
	cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);
	cudaFree(devGray);
   
}


// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0,  1,  2,   1,  0,  0,  
                     0, 3,  13, 22,  13, 3,  0,  
                     1, 13, 59, 97,  59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97,  59, 13, 1,  
                     0, 3,  13, 22,  13, 3,  0,
                     0, 0,  1,  2,   1,  0,  0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

// Kernel for lab 5 WITH SHARED MEMODY
__global__ void gaussianBlurShared (uchar3 * input, uchar3 * output, int * weight, int imageWidth, int imageHeight, int blurDim) {
	
	__shared__ int blurMatrix[49] ;
	
	if (threadIdx.x <= 49) {
		blurMatrix[threadIdx.x] = weight[threadIdx.x] ;
	}

	__syncthreads() ;
	
	int tidx = threadIdx.x + blockIdx.x * blockDim.x ;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y ;
	int originTid = tidx +  imageWidth *  tidy ;// getting the center pixel 
	int relativTid;

	// if the pixel is in the pixel 
	if( tidx >= imageWidth || tidy >= imageHeight) return ;
	
	// sum of the pixel (weight * value) and coeficient 
	int sum = 0 ;
	int coef = 0 ;
	
	// Process the 9 pixels 
	for (int i = -(blurDim) ; i < (blurDim) ; i++){
		for (int j = -(blurDim) ; j < (blurDim) ; j++) {
			
			// getting the relative position of our relativPixel in X and Y
			int x = tidx + i ;
			int y = tidy + j ;
			
			// Checking if it is not out of bounds...
			if (x >= imageWidth  || x < 0) continue ;
			if (y >= imageHeight || y < 0) continue ;

			// working on a specific pixel relative to the threaded pixel
			relativTid = imageWidth * y + x ;
			
			// applying the blur on gray pixel 
			unsigned char gray = (input[relativTid].x + input[relativTid].y + input[relativTid].z) /3;
            int coefficient = blurMatrix[(j+(blurDim)) * 7 + (i+(blurDim))];
            sum  = sum + gray * coefficient ;
            coef += coefficient;
		}
	}
	sum /= coef;
	output[originTid].x = output[originTid].y = output[originTid].z = sum ;
}


// USING SHARED MEMORY KERNELs
void Labwork::labwork5_GPU(int blockNumber, int blurDim) {
	
	int blurMatrix[] = {0, 0,  1,  2,   1,  0,  0,  
                     	0, 3,  13, 22,  13, 3,  0,  
                     	1, 13, 59, 97,  59, 13, 1,  
                     	2, 22, 97, 159, 97, 22, 2,  
                     	1, 13, 59, 97,  59, 13, 1,  
                     	0, 3,  13, 22,  13, 3,  0,
                     	0, 0,  1,  2,   1,  0,  0 };
	
	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;

	
	// We set grid size and block size as dim3 variables
	dim3 gridSize = dim3(inputImage->width / blockNumber, inputImage->height / blockNumber);
	dim3 blockSize2 = dim3(blockNumber, blockNumber);
	

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 
	uchar3 * devBlur ;
	int * devWeight ;
	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devBlur, pixelCount * sizeof(uchar3));
	cudaMalloc(&devWeight, sizeof(blurMatrix)); 
	
	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(devWeight, blurMatrix, sizeof(blurMatrix), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	gaussianBlurShared<<<gridSize, blockSize2>>>(devInput, devBlur, devWeight, inputImage->width, inputImage->height, blurDim) ; 
	
	// Gettting the results from GPU to CPU 
	cudaMemcpy(outputImage, devBlur, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);
	cudaFree(devBlur);
    cudaFree(devWeight) ;
    
}

// kernel USING GLOBAL MEMORY 
__global__ void gaussianBlur (uchar3 * input, uchar3 * output, int imageWidth, int imageHeight, int blurDim) {
	int blurMatrix[] = {0, 0,  1,  2,   1,  0,  0,  
                     	0, 3,  13, 22,  13, 3,  0,  
                     	1, 13, 59, 97,  59, 13, 1,  
                     	2, 22, 97, 159, 97, 22, 2,  
                     	1, 13, 59, 97,  59, 13, 1,  
                     	0, 3,  13, 22,  13, 3,  0,
                     	0, 0,  1,  2,   1,  0,  0 };
	
	// getting the pixel location 
	int tidx = threadIdx.x + blockIdx.x * blockDim.x ;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y ;
	int originTid = tidx +  imageWidth *  tidy ;// getting the center pixel 	
	int relativTid; // this px is nearby the center pixel

	// if the center pixel is not out of bound 
	if( tidx >= imageWidth || tidy >= imageHeight) return ;
	
	/* ALGORITHM IMPLEMENTATION */
	// sum of the pixel (weight * value) and coeficient 
	int sum = 0 ;
	int coef = 0 ;
	
	// Process the 9 pixels 
	for (int i = -(blurDim) ; i < (blurDim) ; i++){
		for (int j = -(blurDim) ; j < (blurDim) ; j++) {
			
			// getting the relative position of our relativPixel in X and Y
			int x = tidx + i ;
			int y = tidy + j ;
			
			// Checking if it is not out of bounds...
			if (x >= imageWidth  || x < 0) continue ;
			if (y >= imageHeight || y < 0) continue ;

			// working on a specific pixel relative to the threaded pixel
			relativTid = imageWidth * y + x ;
			
			// applying the blur on gray pixel 
			unsigned char gray = (input[relativTid].x + input[relativTid].y + input[relativTid].z) /3;
            int coefficient = blurMatrix[(j+(blurDim)) * 7 + (i+(blurDim))];
            sum  = sum + gray * coefficient ;
            coef += coefficient;
		}
	}
	sum /= coef;
	output[originTid].x = output[originTid].y = output[originTid].z = sum ;
}

// lab 5 USING GLOBAL MEMORY
void Labwork::labwork5_GPU_not_shared(int blockNumber, int blurDim) {
	
	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;

	
	// We set grid size and block size as dim3 variables
	dim3 gridSize = dim3(inputImage->width / blockNumber, inputImage->height / blockNumber);
	dim3 blockSize2 = dim3(blockNumber, blockNumber);
	

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 
	uchar3 * devBlur ;
	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devBlur, pixelCount * sizeof(uchar3));
	
	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	gaussianBlur<<<gridSize, blockSize2>>>(devInput, devBlur, inputImage->width, inputImage->height, blurDim) ; 
	
	// Gettting the results from GPU to CPU 
	cudaMemcpy(outputImage, devBlur, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);
	cudaFree(devBlur);
    
}


// Kernel for lab 6a 
__global__ void binarization(uchar3 *input, uchar3 *output, int imageWidth, int imageHeight) {
	
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;
	
	// Out of bound verification
	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	
	int tid = tidx + imageWidth * tidy ;
	
	// integer division avoid conditions
	output[tid].x = (int) ((input[tid].x ) / 128) * 255;
	output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork6_GPU_binarization(int blockNumber) {

	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;

	
	// We set grid size and block size as dim3 variables
	dim3 gridSize = dim3(inputImage->width / blockNumber, inputImage->height / blockNumber);
	dim3 blockSize2 = dim3(blockNumber, blockNumber);
	

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 
	uchar3 * devBinarize ;
	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devBinarize, pixelCount * sizeof(uchar3));
	
	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	binarization<<<gridSize, blockSize2>>>(devInput, devBinarize, inputImage->width, inputImage->height) ; 
	
	// Gettting the results from GPU to CPU 
	cudaMemcpy(outputImage, devBinarize, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);
	cudaFree(devBinarize);
}

// Kernel for lab 6b
__global__ void brightness(uchar3 *input, uchar3 *output, int imageWidth, int imageHeight, int brightnessValue) {
	
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;

	if (tidx >= imageWidth || tidy >= imageHeight) return ;

	int tid = tidx + imageWidth * tidy ;

	output[tid].z = max(min(input[tid].z + brightnessValue, 255), 0); 
	output[tid].y = max(min(input[tid].y + brightnessValue, 255), 0); 
	output[tid].x = max(min(input[tid].x + brightnessValue, 255), 0); 
}

void Labwork::labwork6_GPU_brightness(int blockNumber, int brightnessValue){
	
	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;

	
	// We set grid size and block size as dim3 variables
	dim3 gridSize = dim3(inputImage->width / blockNumber, inputImage->height / blockNumber);
	dim3 blockSize2 = dim3(blockNumber, blockNumber);
	

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 
	uchar3 * devBrighted ;
	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devBrighted, pixelCount * sizeof(uchar3));
	
	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	brightness<<<gridSize, blockSize2>>>(devInput, devBrighted, inputImage->width, inputImage->height, brightnessValue) ; 
	
	// Gettting the results from GPU to CPU 
	cudaMemcpy(outputImage, devBrighted, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);
	cudaFree(devBrighted);
}

// Kernel for lab 6c
__global__ void blending(uchar3 *input, uchar3 *output, int imageWidth, int imageHeight, uchar3 * inputImage2, float blendCoeficient) {
	
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;

	int tid = tidx + imageWidth * tidy ;
	
	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	
	output[tid].z = blendCoeficient * input[tid].z + (1 - blendCoeficient) * inputImage2[tid].z ;
	output[tid].y = blendCoeficient * input[tid].y + (1 - blendCoeficient) * inputImage2[tid].y ;
	output[tid].x = blendCoeficient * input[tid].x + (1 - blendCoeficient) * inputImage2[tid].x ;
	
}
void Labwork::labwork6_GPU_blending(int blockNumber, JpegInfo * inputImage2, float blendCoeficient) {
	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;


	// We set grid size and block size as dim3 variables
	dim3 gridSize = dim3(inputImage->width / blockNumber, inputImage->height / blockNumber);
	dim3 blockSize2 = dim3(blockNumber, blockNumber);


	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 
	uchar3 * devBlend ;
	uchar3 * devInput2; 

	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devBlend, pixelCount * sizeof(uchar3));
	cudaMalloc(&devInput2,   pixelCount * sizeof(uchar3));

	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput,  inputImage->buffer,  pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(devInput2, inputImage2->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

	// Using the kernel with the dim3 block and grid size
	blending<<<gridSize, blockSize2>>>(devInput, devBlend, inputImage->width, inputImage->height, devInput2, blendCoeficient) ; 

	// Gettting the results from GPU to CPU 
	cudaMemcpy(outputImage, devBlend, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

	// FREEEE
	cudaFree(devInput);
	cudaFree(devBlend);
	cudaFree(devInput2);
}


// Kernel for lab 7 : reduce the image into two arrays containing the min and max of each of treated blocks 
__global__ void reduceImage(uchar3 * in, int * outMin, int * outMax, int imageWidth, int imageHeight) {
	
	// getting the cache ID
	int localtid = threadIdx.x; 
	// getting the global ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid > imageWidth * imageHeight) return ;
	
	// Make a cache array that contains the <BlockSize> value of gray
	__shared__ int cacheMin[1024] ;
	__shared__ int cacheMax[1024] ;	
	
	// caching the gray values
	cacheMin[localtid] = (in[tid].x + in[tid].y + in[tid].z)/3;
	cacheMax[localtid] = (in[tid].x + in[tid].y + in[tid].z)/3;
	
	__syncthreads();
	

	// reduction in cache
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		// only processing the half lower values of the array
		if (localtid < s) {
			// compare 2 by 2
			cacheMin[localtid] = min(cacheMin[localtid], cacheMin[localtid + s]);
			cacheMax[localtid] = max(cacheMax[localtid], cacheMax[localtid + s]);
		}
		__syncthreads();
	}	
	
	
	if (localtid == 0){ 
		outMin[blockIdx.x] = cacheMin[0];
		outMax[blockIdx.x] = cacheMax[0];
	}
}

__global__ void reduceMin(int * in, int * out, int imageWidth, int imageHeight) {	
	unsigned int localtid = threadIdx.x; // local id 
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;// global id 
	if (tid > imageWidth) return ;
	
	__shared__ int cache[1024] ;
	cache[localtid] = in[tid];
	__syncthreads();
	
	// reduction in cache
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		if (localtid < s) {
			cache[localtid] = min(cache[localtid], cache[localtid + s]);
		}
		__syncthreads();
	}		
	
	if (localtid == 0){ 
		out[blockIdx.x] = cache[0]; 
	}
}

__global__ void reduceMax(int * in, int * out, int imageWidth, int imageHeight) {
	__shared__ int cache[1024] ;
	// cache the block content
	unsigned int localtid = threadIdx.x;
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid > imageWidth) return ;
	cache[localtid] = in[tid];
	__syncthreads();
	
	// reduction in cache
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		
		if (localtid < s) {
			cache[localtid] = max(cache[localtid], cache[localtid + s]);
		}
		__syncthreads();
	}		
	
	if (localtid == 0){ 
		out[blockIdx.x] = cache[0]; 
	}
}

// stretching kernel for daily exercises 
__global__ void stretch (uchar3 * devInput, uchar3 * devOutput, int imageWidth, int imageHeight, int minValue, int maxValue) {
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;

	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	int tid = tidx + imageWidth * tidy ;
	
	/*IMPLEMENTATION*/
	devOutput[tid].x = (double(devInput[tid].x - minValue) / double(maxValue-minValue))*255;
	devOutput[tid].y = (double(devInput[tid].y - minValue) / double(maxValue-minValue))*255;
	devOutput[tid].z = (double(devInput[tid].z - minValue) / double(maxValue-minValue))*255;

}

void Labwork::labwork7_GPU(int blockNumber) {
	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;
	// int blockSize = atoi(argv[3]); // replace by the parameter
	int blockSize2 = 1024 ;
	int gridSize = pixelCount/blockSize2 ;

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output)
	int * mins = (int *) calloc(gridSize, sizeof(int)) ;
	int * maxs = (int *) calloc(gridSize, sizeof(int)) ;
	uchar3 * devInput ; 
	uchar3 * devOutput ;
	int * devMin ;
	int * devMax ; 
	
	cudaMalloc(&devInput,  pixelCount * sizeof(uchar3));
	cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devMin, gridSize * sizeof(int));
	cudaMalloc(&devMax, gridSize * sizeof(int));

	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel to convert in gray AND find a minimum by block
	reduceImage<<<gridSize, blockSize2>>>(devInput, devMin, devMax, inputImage->width, inputImage->height) ; 

	while(gridSize > 1) {
		reduceMin<<<gridSize, blockSize2>>>(devMin, devMin, inputImage->width, inputImage->height) ;
		reduceMax<<<gridSize, blockSize2>>>(devMax, devMax, inputImage->width, inputImage->height) ;
		gridSize = max(gridSize/blockSize2, 1);
	}

	// Gettting the results from GPU to CPU 
	cudaMemcpy(mins, devMin, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(maxs, devMax, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

	//stretch the image 
	dim3 blockSize2D = dim3(blockNumber,blockNumber);
    	dim3 gridSize2D = dim3(ceil(inputImage->width/blockSize2D.x), ceil(inputImage->height/blockSize2D.y));
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	stretch<<<gridSize2D, blockSize2D>>>(devInput, devOutput, inputImage->width, inputImage->height, mins[0], maxs[0]) ;
	
	// preparing the output
        outputImage = static_cast<char *>(malloc(pixelCount * 3));
	cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);
	cudaFree(devOutput);
	cudaFree(devMin);
	free(mins);
	free(maxs);
}

// kernel for 8 with maths optimisation

__device__ int sign(bool x) {
	// return 1 if > 0, 0 if 0, -1 if < 0
	return (x > 0) - (x < 0) ;
}
__global__ void RGB2HSVMaths(uchar3 *input, int imageWidth, int imageHeight, float * h, float *s, float *v) {
/*	
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;

	int tid = tidx + imageWidth * tidy ;
	
	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	
	// floating rgb values
	float R = (float) input[tid].x /255 ;
	float G = (float) input[tid].y /255 ;
	float B = (float) input[tid].z /255 ;

	// Min and max from our pixel, then the delta
	float deltaMin = min(min(R, G), B) ;
	float deltaMax = max(max(R, G), B) ;

	float delta = deltaMax - deltaMin  ;
	
	
	// making XYZ matrix to prepare the final formula 
	bool maxColor[3] ; 
	maxColor[0] = R != deltaMax ;
	maxColor[1] = R == deltaMax || G != deltaMax ;
	maxColor[2] = R == deltaMax || G == deltaMax ;
	if (delta == 0.0 || deltaMax == 0) {
		h[tid] = 0 ;
		s[tid] = 0 ;		
		v[tid] = 0 ;
	}
	
	// Mathematical methode based on this paper 
	// http://www.daaam.info/Downloads/Pdfs/proceedings/proceedings_2011/1591_Kobalicek.pdf
	// Aborted because of it computing time ~80ms

	// define H 
	float hbase = (-(4/6) && maxColor[0] ^ sign(!maxColor[2])) + 1.0 ;
	float Rm = (R && maxColor[0]) ^ sign(!maxColor[1]) ;
	float Gm = (G && maxColor[1]) ^ sign(!maxColor[2]) ;
	float Bm = (B && maxColor[2]) ^ sign( maxColor[1]) ;

	// TODO : Try to debug having the first pixel of eiffel.jpg at H=3,79
	float H = fmod((hbase + (Rm + Gm + Bm)/(6*delta)), (float)1);
	h[tid] = H;
	s[tid] = delta/deltaMax;
	v[tid] = deltaMax;
*/	
}
void Labwork::labwork8_GPU_maths() {
	/*// useful variables 
	int pixelCount = inputImage->width * inputImage->height;

	
	// We set grid size and block size as dim3 variables
	dim3 blockSize = dim3(32,32);
	dim3 gridSize = dim3(ceil(inputImage->width/blockSize.x), ceil(inputImage->height/blockSize.y));

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 


	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	//cudaMalloc(&devBrighted, pixelCount * sizeof(uchar3));
	cudaMalloc(&outH, pixelCount * sizeof(float));
	cudaMalloc(&outS, pixelCount * sizeof(float));
	cudaMalloc(&outV, pixelCount * sizeof(float));
	
	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	RGB2HSVMaths<<<gridSize, blockSize>>>(devInput, inputImage->width, inputImage->height, outH, outS, outV) ;
	
	
	// Gettting the results from GPU to CPU 
	
	
	
	// FREEEE
	cudaFree(devInput);

	free(H);
	free(S);
	free(V);
*/
}
typedef struct _HSV {
	float * H ;
	float * S ;
	float * V ;
} HSV;
// Lab 8 with scatter
__global__ void RGB2HSV(uchar3 *input, int imageWidth, int imageHeight, HSV hsv) {
	
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;
	int tid = tidx + imageWidth * tidy ;
	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	
	// floating rgb values
	float R = (float) input[tid].x /255.0 ;
	float G = (float) input[tid].y /255.0 ;
	float B = (float) input[tid].z /255.0 ;

	// Finding Min and max from our R, G or B value of our pixel
	float deltaMin = min(min(R, G), B) ;
	float deltaMax = max(max(R, G), B) ;
	float delta = deltaMax - deltaMin ;
	
	// Delta, also called V, can be set in our structure
	hsv.V[tid] = deltaMax ;
	
	// worst case scenario 
	if (deltaMax == 0 ) {
		hsv.H[tid] = 0 ;
		hsv.S[tid] = 0 ;
		return ;
	}
	
	// S can be set in our structure 
	hsv.S[tid] = (float) (delta/deltaMax);
	
	if (delta == 0) {
		hsv.H[tid] = 0 ; 
		return ;
	}


	// DEFINING H (in a silly way) ...
	if (R >= deltaMax) {
		hsv.H[tid] = fmodf((G - B)/delta, 6.0) ; 
		return ;
	}
	
	if (G >= deltaMax) {
		hsv.H[tid] = (2.0 + (B - R)/delta) ;
		return ;
	}

	// default case 
	if (B >= deltaMax) {
		hsv.H[tid] =(4.0 + (R - G)/delta) ;
		return  ;
	}

	
}
__global__ void HSV2RGB(HSV hsv, int imageWidth, int imageHeight, uchar3 *output) {
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;

	int tid = tidx + imageWidth * tidy ;
	
	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	
	float h = hsv.H[tid] ;
	float s = hsv.S[tid] ;
	float v = hsv.V[tid] ; // *255 easing futur multiplication 
	
	
	float d = h / 60;
	float hi = ((int) d % 6); // skippable
	float f = d - hi;
    
	float l = v * (1.0 - s);
	float m = v * (1.0 - f * s);
	float n = v * (1.0 - ( 1.0 - f ) * s);
	
	if(h < M_PI/3) {     
	    output[tid].x = v * 255 ;
	    output[tid].y = n * 255 ;
	    output[tid].z = l * 255 ;
	    return;   
	}
   
	if(h >= M_PI/3 && h < M_PI*2/3) {     
	    output[tid].x = m * 255 ;
	    output[tid].y = v * 255 ;
	    output[tid].z = l * 255 ;
	    return;
	}
   
   
	if(h >= M_PI*2/3 && h < M_PI) {     
	    output[tid].x = l * 255 ;
	    output[tid].y = v * 255 ;
	    output[tid].z = n * 255 ;
	    return;
	}
   
   
	if(h >= M_PI && h < M_PI*4/3) {     
	    output[tid].x = l * 255 ;
	    output[tid].y = m * 255 ;
	    output[tid].z = v * 255 ;
	    return;
	}
   
   
	if(h >= M_PI*4/3 && h < M_PI*5/3) {     
	    output[tid].x = n * 255 ;
	    output[tid].y = l * 255 ;
	    output[tid].z = v * 255 ;
	    return;
	}
       
	if(h >= M_PI*5/3) {
	    output[tid].x = v * 255 ;
	    output[tid].y = l * 255 ;
	    output[tid].z = m * 255 ;
	    return;
	}
}

void Labwork::labwork8_GPU_scatter(int blockNumber) {
	// useful variables 
	int pixelCount = inputImage->width * inputImage->height;

	
	// We set grid size and block size as dim3 variables
	dim3 blockSize = dim3(blockNumber,blockNumber);
	dim3 gridSize = dim3(ceil(inputImage->width/blockNumber), ceil(inputImage->height/blockNumber));

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 
	uchar3 * devOut ;
	
	HSV hsv ;

	cudaMalloc(&devOut, pixelCount * sizeof(uchar3));	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc((void **) &hsv.H, pixelCount * sizeof(float));
	cudaMalloc((void **) &hsv.S, pixelCount * sizeof(float));
	cudaMalloc((void **) &hsv.V, pixelCount * sizeof(float));
	
	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	RGB2HSV<<<gridSize, blockSize>>>(devInput, inputImage->width, inputImage->height, hsv) ; 
	HSV2RGB<<<gridSize, blockSize>>>(hsv, inputImage->width, inputImage->height, devOut) ; 
	
	cudaMemcpy(outputImage, devOut, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);	
	cudaFree(hsv.H);
	cudaFree(hsv.S);
	cudaFree(hsv.V);
	cudaFree(devOut);
}

__global__ void histogramize(uchar3 *input, int imageWidth, int imageHeight, int *histogram) {

	/* VERIFICATION */		
	int localtid = threadIdx.x ;
	int tid = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int pixelCount = imageWidth * imageHeight;
		
	if (tid > imageWidth * imageHeight) return ;

	/* IMPLEMENTATION */
	// Caching the gray values to shared memory for the first thread 
	__shared__ int cacheHist[1024] ;	
	cacheHist[threadIdx.x] = (input[tid].x 
				+ input[tid].y 
				+ input[tid].z) / 3;
	__syncthreads();
	
	// only the first thread of the block is allowed to copy the shared memory to the output
	if (localtid == 0){
		for (int i = 0 ; (i < 1024) && (tid + i < pixelCount) ; i++) {
			histogram[ blockIdx.x * 256 + cacheHist[i] ] ++;
		}
	}
}

__global__ void histogramReduction(int * inHists, int pixelCount, int endOfArrays ) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;	
	int localtid = threadIdx.x ;
	int bid = blockIdx.x ;
	
	if (tid > pixelCount) return ;
	
	// set up the 4 histogram in good places
	int histIntensity = localtid % 256;
	int histNumber = bid * 256 ; 
	int firstValueOfHist = int(localtid/256) * gridDim.x * 256;
	int localIndex = histNumber + histIntesity + firstValueOfHist; 
	
	if (localIndex > endOfArrays) return ;
	
	// store the histogram value in the right cache place 
	__shared__ int cache[1024];
	cache[localtid]  = inHists[localIndex] ;
	__syncthreads();

	// calculate the number of histogram to sum (for future optimisations)
	int step = 256 ;  // the step between two histograms 
	int step_nb = 3 ; // the number of histogram to sum
	if (localtid < 256)	{
		for (int i = 1 ; i <= step_nb ; i++) {
			cache[localtid] += cache[localtid + step*i] ;
		}
		// Setting the histogram value to the right place in output array 
		inHists[localtid + bid *256] = cache[localtid] ;
	}
}

__global__ void histogramStrecth(uchar3 * input, uchar3 * output, int * histogram, int pixelCount) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x ; 
	if (tid > pixelCount) return ;

	__shared__ float cacheHisto[256] ; 
	__shared__ float n; // (patchwork) the total of pixel in the histogram 
	if (threadIdx.x < 256) cacheHisto[threadIdx.x] = histogram[threadIdx.x]; 
	n = 0.0 ;
	__syncthreads();
	
	if (threadIdx.x == 0){
		// (patchwork : calculate n)
		for (int i = 0 ; i < 256 ; i++) {
			n += (float)cacheHisto[i];
		}
		
		// calculate the sum of intensities proportions in cacheHisto 
		cacheHisto[0] = float(cacheHisto[0]) / float(n) ;
		for (int i = 1 ; i < 256 ; i++) {
			cacheHisto[i] = float(cacheHisto[i] / n) + cacheHisto[i-1]	;
		}
	}
	__syncthreads();
	
	int grayValue = (input[tid].x + input[tid].y + input[tid].z) / 3 ;
	// converting the in value in stretched histogram value	
	output[tid].x = output[tid].y = output[tid].z = int(cacheHisto[grayValue] * 255);
	
}

void Labwork::labwork9_GPU() {

	// We set grid size and block size as dim3 variables
	int pixelCount = inputImage->width * inputImage->height;

	int blockSize =1024;
	int gridSize = ceil((float)pixelCount / (float)blockSize) ;

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 
	uchar3 * devOut ;
	
	// access an hist array -> histogram[localtid + rgbValue] 
	int * histogram ;
	int * histogramOut = (int*) malloc(gridSize * 256 * sizeof(int));

	cudaMalloc(&devOut, pixelCount * sizeof(uchar3));	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&histogram, gridSize * sizeof(int) * 256);
	
	
	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	histogramize<<<gridSize, blockSize>>>(devInput, inputImage->width, inputImage->height, histogram) ; 
	
	// The endOfArrays shows the last value to care in our giant histogram
	int endOfArrays = gridSize * 256 - 1	   ;
	do {
		// We treat 4 histogram at the same time with our 1024 block-sized kernel
		gridSize = int( ceil( float(gridSize)/4.0 ) ) ;
		histogramReduction<<<gridSize, blockSize>>>(histogram, pixelCount, endOfArrays) ;
		endOfArrays = gridSize * 256 - 1 ;
	
	} while (gridSize > 1);
	
	// pixel number in the end of reduction (should be pixelCount if the reduction worked... but you know...)
	gridSize = ceil((float)pixelCount / (float)blockSize) ;
	histogramStrecth<<<gridSize, blockSize>>>(devInput, devOut, histogram, pixelCount);
		
	cudaMemcpy(outputImage, devOut, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost) ;
		
	// FREEEE
	cudaFree(devInput);	
	cudaFree(histogram);
	free(histogramOut);
	cudaFree(devOut);
}

__device__ void add(float * sumrgbv, uchar3 inPixel, float V) {
	sumrgbv[0] += float(inPixel.x) ; 
	sumrgbv[1] += float(inPixel.y) ; 
	sumrgbv[2] += float(inPixel.z) ; 
	sumrgbv[3] += float(V) ;	
}

__global__ void kuwahara(uchar3 * input, int width, int height, HSV hsv, uchar3 * output, int windowSize){
	
	int tidx = threadIdx.x + blockIdx.x * blockDim.x ; 
	int tidy = threadIdx.y + blockIdx.y * blockDim.y ; 

	if (tidx > width || tidy > height) return ;
	if (tidx + windowSize > width || tidx - windowSize < 0) return ;
	if (tidy + windowSize > height || tidy - windowSize < 0) return ;
	
	int tid = tidx + width * tidy ;
	
	// Tabs to avoid mutiple variables 
	float sumrgbvA[4] = {0,0,0,0};
	float sumrgbvB[4] = {0,0,0,0};
	float sumrgbvC[4] = {0,0,0,0};
	float sumrgbvD[4] = {0,0,0,0};

	// summing the windows 
	for (int i = -windowSize ; i <= windowSize ; i++) {
		for (int j = -windowSize ; j <= windowSize ; j++) {

			// getting the relativ pixel Tid
			int pixelIndex = (tidx + i) + width * (tidy + j) ;

			if (i < 1 && j < 1) { // top left window
				add(sumrgbvA, input[pixelIndex], hsv.V[pixelIndex]) ;
			}
			if (i >= 0 && j < 1) { // top right window 
				add(sumrgbvB, input[pixelIndex], hsv.V[pixelIndex]) ;
			}
			if (i < 1 && j >= 0) { // bottom left window
				add(sumrgbvC, input[pixelIndex], hsv.V[pixelIndex]) ;
			}
			if (i >= 0 && j >= 0) { // bottom right window 
				add(sumrgbvD, input[pixelIndex], hsv.V[pixelIndex]) ;
			}			
		}
	}

	
	
	// having the averages in our tabs
	for (int i = 0 ; i < 4 ; i++) {
		sumrgbvA[i] /= (windowSize+1) * (windowSize+1) ;
		sumrgbvB[i] /= (windowSize+1) * (windowSize+1) ;
		sumrgbvC[i] /= (windowSize+1) * (windowSize+1) ;
		sumrgbvD[i] /= (windowSize+1) * (windowSize+1) ;
	}

	
	// calculating the Standard deviation sum 
	float sumSDABCD[4] = {0,0,0,0};

	for (int i = -windowSize ; i <= windowSize ; i++) {
		for (int j = -windowSize ; j <= windowSize ; j++) {

			int pixelIndex = (tidx + i) + width * (tidy + j) ;

			if (i < 1 && j < 1) {
				sumSDABCD[0] += (hsv.V[pixelIndex]-sumrgbvA[3]) * (hsv.V[pixelIndex]-sumrgbvA[3]);
			}
			if (i >= 0 && j < 1) {
				sumSDABCD[1] += (hsv.V[pixelIndex]-sumrgbvB[3]) * (hsv.V[pixelIndex]-sumrgbvB[3]);
			}
			if (i < 1 && j >= 0) {
				sumSDABCD[2] += (hsv.V[pixelIndex]-sumrgbvC[3]) * (hsv.V[pixelIndex]-sumrgbvC[3]);
			}
			if (i >= 0 && j >= 0) {
				sumSDABCD[3] += (hsv.V[pixelIndex]-sumrgbvD[3]) * (hsv.V[pixelIndex]-sumrgbvD[3]);
			}
		}
	}
	
	
	// calculating standard deviation 
	for (int i = 0 ; i < 4 ; i++) {
		sumSDABCD[i] = (sumSDABCD[i]) / ((windowSize + 1) *(windowSize + 1)) ;
	}
	
	// finding the minimum of brightness SD and which one is it
	float minSD = min( min( min( sumSDABCD[0] , sumSDABCD[1] ), sumSDABCD[2] ), sumSDABCD[3] ) ;
	// Which one is it ?
	bool minSDTab[4] = {
		sumSDABCD[0] == minSD, 
		sumSDABCD[1] == minSD,
		sumSDABCD[2] == minSD,
		sumSDABCD[3] == minSD
	} ;
	// How much are they ? 
	int sum_sum = minSDTab[0] + minSDTab[1] + minSDTab[2] + minSDTab[3];

	// only the minimum of SD will have a *1 multiplication, the others products are 0
	output[tid].x = (minSDTab[0] * sumrgbvA[0]
			  + minSDTab[1] * sumrgbvB[0] 
			  + minSDTab[2] * sumrgbvC[0] 
			  + minSDTab[3] * sumrgbvD[0]) / sum_sum ; 
				  
	output[tid].y = (minSDTab[0] * sumrgbvA[1]
			  + minSDTab[1] * sumrgbvB[1] 
			  + minSDTab[2] * sumrgbvC[1] 
			  + minSDTab[3] * sumrgbvD[1]) / sum_sum ; 
			  
	output[tid].z = (minSDTab[0] * sumrgbvA[2]
			  + minSDTab[1] * sumrgbvB[2] 
			  + minSDTab[2] * sumrgbvC[2] 
			  + minSDTab[3] * sumrgbvD[2]) / sum_sum ; 
}

void Labwork::labwork10_GPU(int kuwaharaCoef) {
	// useful variables 
	
	int pixelCount = inputImage->width * inputImage->height;	
	
	// We set grid size and block size as dim3 variables
	dim3 blockSize = dim3(32,32);
	dim3 gridSize = dim3(ceil(inputImage->width/blockSize.x), ceil(inputImage->height/blockSize.y));

	// Allocating the output image 
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	// Allocating the device memory for the image (input and output) and weight matrix
	uchar3 * devInput ; 
	uchar3 * devOut ;
	
	HSV hsv ;

	cudaMalloc(&devOut, pixelCount * sizeof(uchar3));	
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc((void **) &hsv.H, pixelCount * sizeof(float));
	cudaMalloc((void **) &hsv.S, pixelCount * sizeof(float));
	cudaMalloc((void **) &hsv.V, pixelCount * sizeof(float));
	
	// Copying the data from CPU to GPU 
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	// Using the kernel with the dim3 block and grid size
	RGB2HSV<<<gridSize, blockSize>>>(devInput, inputImage->width, inputImage->height, hsv) ; 
	kuwahara<<<gridSize, blockSize>>>(devInput, inputImage->width, inputImage->height, hsv, devOut, kuwaharaCoef) ;
	
	cudaMemcpy(outputImage, devOut, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);	
	cudaFree(hsv.H);
	cudaFree(hsv.S);
	cudaFree(hsv.V);
	cudaFree(devOut);
}


 
