GPUMatrixMultiplicationTest
=====================

GPU matrix multiplication in Unity.

## Usage

float[,] A = new float[1024, 512];
float[,] B = new float[512, 256];

// matmul : ComputeShader (MatMul.compute)
float[,] C = GPUMatrixMultiplication.Multiply(matmul, A, B);

// float[,] C = GPUMatrixMultiplication.Multiply(matmul, A, B, GPUMatrixMultiplicationMethod.SharedMemory); // default
// float[,] C = GPUMatrixMultiplication.Multiply(matmul, A, B, GPUMatrixMultiplicationMethod.Naive); // naive impl


## Resourses

- UC San Diego Lecture Slide - Matrix Multiplication Using Shared Memory - http://cseweb.ucsd.edu/classes/wi12/cse260-a/Lectures/Lec08.pdf
- CUDA in Two-dimension - http://selkie.macalester.edu/csinparallel/modules/GPUProgramming/build/html/CUDA2D/CUDA2D.html


## Compatibility

tested on Unity 2017.0.3, Windows 10 (GTX 1060).
