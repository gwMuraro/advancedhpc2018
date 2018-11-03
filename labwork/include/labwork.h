#pragma once

#include <include/jpegloader.h>
#include <include/timer.h>

class Labwork {

private:
    JpegLoader jpegLoader;
    JpegInfo *inputImage;
    char *outputImage;

public:
    void loadInputImage(std::string inputFileName);
    void saveOutputImage(std::string outputFileName);

    void labwork1_CPU();
    void labwork1_OpenMP();
    void labwork1_OpenMP_doublePragma();

    void labwork2_GPU();

    void labwork3_GPU(int blockSize);

    void labwork4_GPU(int blockSize);

    void labwork5_CPU();
    void labwork5_GPU(int blockNumber, int blurDim);
    void labwork5_GPU_not_shared(int blockNumber, int blurDim) ;

    void labwork6_GPU_binarization(int blockNumber);
    void labwork6_GPU_brightness(int blockNumber, int brightnessValue);
    void labwork6_GPU_blending(int blockNumber);
    void labwork7_GPU();

    void labwork8_GPU();

    void labwork9_GPU();

    void labwork10_GPU();
};
