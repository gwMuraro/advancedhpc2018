#include <stdio.h>
#include <stdlib.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <include/jpegloader.h>

#define ACTIVE_THREADS 4

JpegInfo *inputImage2 ;
Histogram * hist ;

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
	int blurDim = 3 ; // TODO : non-working default value : fix dat :'(
	int brightness = (argv[4] != NULL && argv[4] != 0) ? atoi(argv[4]) : 100;

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
	if (lwNum == 7)	{
		hist = new Histogram() ;
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
            labwork.labwork7_GPU(32, hist->tab);
            
			//printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());            
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
        
        	timer.start();
            labwork.labwork8_GPU_maths();
			printf("labwork %d with maths ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());            
            
        	timer.start();
            labwork.labwork8_GPU_scatter();
			printf("labwork %d with scatter ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());            
			

			labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
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
	
	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	
	int tid = tidx + imageWidth * tidy ;
	
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
	__shared__ int cacheMin[1024] ;
	__shared__ int cacheMax[1024] ;
	// cache the block content
	int localtid = threadIdx.x;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid > imageWidth * imageHeight) return ;

	cacheMin[localtid] = (in[tid].x + in[tid].y + in[tid].z)/3;
	cacheMax[localtid] = (in[tid].x + in[tid].y + in[tid].z)/3;
	
	__syncthreads();
	

	// reduction in cache
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		if (localtid < s) {
			cacheMin[localtid] = min(cacheMin[localtid], cacheMin[localtid + s]);
			cacheMax[localtid] = max(cacheMax[localtid], cacheMax[localtid + s]);
		}
		__syncthreads();
	}	
	
	
	if (localtid == 0){ 
		outMin[blockIdx.x] = cacheMin[0];
		outMax[blockIdx.x] = cacheMax[0];
		//printf("%d |", in[tid]);
	}
}

__global__ void reduceMin(int * in, int * out, int imageWidth, int imageHeight) {
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
			//printf("%d ; %d | ", cache[localtid], cache[localtid+s]) ;
			cache[localtid] = min(cache[localtid], cache[localtid + s]);
		}
		__syncthreads();
	}		
	
	if (localtid == 0){ 
		out[blockIdx.x] = cache[0]; 
		//printf("%d |", out[blockIdx.x]);
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
			//printf("%d ; %d | ", cache[localtid], cache[localtid+s]) ;
			cache[localtid] = max(cache[localtid], cache[localtid + s]);
		}
		__syncthreads();
	}		
	
	if (localtid == 0){ 
		out[blockIdx.x] = cache[0]; 
		//printf("%d |", out[blockIdx.x]);
	}
}

// stretching kernel for daily exercises 
__global__ void stretch (uchar3 * devInput, uchar3 * devOutput, int imageWidth, int imageHeight, int minValue, int maxValue) {
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;

	if (tidx >= imageWidth || tidy >= imageHeight) return ;

	int tid = tidx + imageWidth * tidy ;
	
    devOutput[tid].x = (double(devInput[tid].x - minValue) / double(maxValue-minValue))*255;
	devOutput[tid].y = (double(devInput[tid].y - minValue) / double(maxValue-minValue))*255;
    devOutput[tid].z = (double(devInput[tid].z - minValue) / double(maxValue-minValue))*255;

//	printf("%d ; %d ; %d|| ", devInput[tid], value, devOutput[tid]) ;

}

void Labwork::labwork7_GPU(int blockNumber, int * histogram) {
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
	dim3 blockSize2D = dim3(32,32);
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
	
	
	// making XYZ matrix to know which R G or B is the greatest
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
__global__ void RGB2HSV(uchar3 *input, int imageWidth, int imageHeight, float * H, float * S, float * V) {
	
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;

	int tid = tidx + imageWidth * tidy ;
	
	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	
	// floating rgb values
	float R = (float) input[tid].x /255.0 ;
	float G = (float) input[tid].y /255.0 ;
	float B = (float) input[tid].z /255.0 ;

	// Min and max from our pixel, then the delta
	float deltaMin = min(min(R, G), B) ;
	float deltaMax = max(max(R, G), B) ;
	float delta = deltaMax - deltaMin ;
	
	// DEFINE V
	V[tid] = deltaMax ;
	
	// worst case scenario 
	if (deltaMax <= 0 || delta == 0) {
		H[tid] = 0 ;
		S[tid] = 0 ;
		return ;
	}
	
	// DEFINE S
	S[tid] = (float) (delta/deltaMax);


	// DEFINE H (in a silly way) ...
	if (R >= deltaMax) {
		H[tid] = fmodf((G - B)/delta, 6.0) ; 
		return ;
	}
	
	if (G >= deltaMax) {
		H[tid] = (2.0 + (B - R)/delta) ;
		return ;
	}

	// default case 
	if (B >= deltaMax) {
		H[tid] =(4.0 + (R - G)/delta) ;
		return  ;
	}

	
}
__global__ void HSV2RGB(float * H, float * S, float * V, int imageWidth, int imageHeight, uchar3 *output) {
	//getting the pixel with the second dimension
	int tidx = (threadIdx.x + blockIdx.x * blockDim.x) ; 
	int tidy = (threadIdx.y + blockIdx.y * blockDim.y) ;

	int tid = tidx + imageWidth * tidy ;
	
	if (tidx >= imageWidth || tidy >= imageHeight) return ;
	

	float h = H[tid] ;
	float s = S[tid] ;
	float v = 255 * V[tid] ; // for futur multiplications
	
	
	float d = h / 60;
    float hi = (int) d % 6; // skippable
    float f = d - hi;
    
    float l = v * (1.0 - s);
    float m = v * (1.0 - f * s);
    float n = v * (1.0 - ( 1.0 - f ) * s);
	
	if(h < M_PI/3) {     
	    output[tid].x = v  ;
	    output[tid].y = n  ;
	    output[tid].z = l  ;
	    return;
   
    }
   
    if(h >= M_PI/3 && h < M_PI*2/3) {     
	    output[tid].x = m  ;
	    output[tid].y = v  ;
	    output[tid].z = l  ;
	    return;
    }
   
   
    if(h >= M_PI*2/3 && h < M_PI) {     
	    output[tid].x = l  ;
	    output[tid].y = v  ;
	    output[tid].z = n  ;
	    return;
    }
   
   
    if(h >= M_PI && h < M_PI*4/3) {     
	    output[tid].x = l  ;
	    output[tid].y = m  ;
	    output[tid].z = v  ;
	    return;
    }
   
   
    if(h >= M_PI*4/3 && h < M_PI*5/3) {     
	    output[tid].x = n  ;
	    output[tid].y = l  ;
	    output[tid].z = v  ;
	    return;
    }
       
    if(h >= M_PI*5/3) {
	    output[tid].x = v  ;
	    output[tid].y = l  ;
	    output[tid].z = m  ;
	    return;
    }
}

void Labwork::labwork8_GPU_scatter() {
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
	RGB2HSV<<<gridSize, blockSize>>>(devInput, inputImage->width, inputImage->height, hsv.H, hsv.S, hsv.V) ; 
	HSV2RGB<<<gridSize, blockSize>>>(hsv.H, hsv.S, hsv.V, inputImage->width, inputImage->height, devOut) ; 
	
	cudaMemcpy(outputImage, devOut, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);	
	cudaFree(hsv.H);
	cudaFree(hsv.S);
	cudaFree(hsv.V);
	cudaFree(devOut);
	

}
void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}



/* HISTOGRAM CLASS */
Histogram::Histogram() {
	tab = (int *) malloc(sizeof(int)*256);
	for (int i = 0 ; i < 256 ; i++) {
		tab[i] = 0 ;
	}
}
size_t Histogram::size() {
	return (sizeof(tab) / sizeof(tab[0])) ;
}

