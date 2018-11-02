#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
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
    int blockSize = (argv[3] != NULL ?atoi(argv[3]):1024) ;
    int blockNumber = (argv[3] != NULL && atoi(argv[3]) < 33) ? atoi(argv[3]) : 32 ;
	int blurDim = (argv[4] != NULL ?atoi(argv[4]):3) ;


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
            timer.start() ;
            labwork.labwork5_GPU(blockNumber, blurDim);
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
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
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
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

// Kernel for lab 5
__global__ void gaussianBlur (uchar3 * input, uchar3 * output, int * weight, int imageWidth, int imageHeight, int blurDim) {
	
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
	gaussianBlur<<<gridSize, blockSize2>>>(devInput, devBlur, devWeight, inputImage->width, inputImage->height, blurDim) ; 
	
	// Gettting the results from GPU to CPU 
	cudaMemcpy(outputImage, devBlur, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	
	// FREEEE
	cudaFree(devInput);
	cudaFree(devBlur);
    cudaFree(devWeight) ;
    
}

void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
