#include <stdio.h>

#define BLOCK_SIZE 32
#define DSIZE 512
#define RADIUS 3

bool debug = false;
bool printouts = true;

// error checking macro
#define cudaCheckErrors()                                       \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Error:  %s at %s:%d \n",           \
			cudaGetErrorString(__err),__FILE__, __LINE__);      \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)


__global__ void stencil_2d(const int *in, int *out, const int length) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];
	int gidx = blockIdx.x * blockDim.x + threadIdx.x;
	int lidx = threadIdx.x + RADIUS;
	int gidy = blockIdx.y * blockDim.y + threadIdx.y;
	int lidy = threadIdx.y + RADIUS;

	// Read input elements into shared memory
	temp[lidx][lidy] = in[gidy + length * gidx];

	if (threadIdx.x < RADIUS) {
		temp[lidx - RADIUS][lidy] = in[gidy + length * (gidx - RADIUS)];
		temp[lidx + BLOCK_SIZE][lidy] = in[gidy + length * (gidx + BLOCK_SIZE)];
	}

	if (threadIdx.y < RADIUS) {
		temp[lidx][lidy - RADIUS] = in[(gidy - RADIUS) + length * gidx];
		temp[lidx][lidy + BLOCK_SIZE] = in[(gidy + BLOCK_SIZE) + length * gidx];
	}

    __syncthreads();

	// Apply the stencil
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++){
		result += temp[lidx + offset][lidy];
		if (offset != 0) {
			result += temp[lidx][lidy + offset];
		}
	}

	// Store the result
	out[gidy + length * gidx] = result;
}


__global__ void dot_product(const int *A, const int *B, int *C, const int length) {	
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y; 

    // Apply dot product
	int result = 0;
    if ((idx < length) && (idy < length)) {
        result = 0;
        for (int i = 0; i < length; i++){
            result += A[idy * length + i] * B[idx + i * length];
        }
        C[idy * length + idx] = result;                    
    }
}


void print_matrix(const int *matrix, const int length) {
	int maxPrintOut = 10;
	for (int i = 0; i < maxPrintOut; i++) {
		for (int j = 0; j < maxPrintOut; j++) {
			printf("%d ", matrix[j + i * length]);
		}
		printf("\n");
	}
}


void check_stencil_2d(const int *in, int *out, const int length) {
	int result;
    for (int i = 0; i < length; i++){
        for (int j = 0; j < length; j++){
            result = 0;
            if (i < RADIUS || i >= length - RADIUS || j < RADIUS || j >= length - RADIUS) {
                result = in[j + i * length];
            }
            else {
                for (int offset = -RADIUS; offset <= RADIUS; offset++) {
		        	result += in[j + (i + offset) * length];
		        	if (offset != 0) {
			        	result += in[(j + offset) + i * length];
		        	}
            	}
			}
            if (result != out[j + i * length]){
                printf("stencil_2d Failed\n");
				printf("Error! at %d, %d. Expected %d, got %d\n", i, j, result, out[j + i * length]);
				exit(1);
            }
        }
    }
	printf("stencil_2d Passed\n");
}


void check_dot_product(const int *A, const int *B, int *C, int length) {	
	int result;
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            result = 0;
            for (int k = 0; k < length; k++){
                result += A[k + i * length] * B[j + k * length];
            }
            if (result != C[j + i * length]){
                printf("dot_product Failed\n");
				printf("Error! at %d, %d. Expected %d, got %d\n", i, j, result, C[j + i * length]);
				exit(1);
            }
        }
    }
	printf("dot_product Passed\n");
}


int main() {
	
	// Create the device and host pointers
	int *h_A, *h_B;
	int *h_A_s, *h_B_s;
	int *h_C;

	// Allocate host memory 
	int length = (DSIZE + 2 * RADIUS);
	int area = length * length * sizeof(int);
	cudaMallocManaged(&h_A, area);
	cudaMallocManaged(&h_B, area);
	cudaMallocManaged(&h_A_s, area);
	cudaMallocManaged(&h_B_s, area);
	cudaMallocManaged(&h_C, area);

	// Fill in the host pointers
	int num = 0;
	for (int i = 0; i < length; i++){
		for (int j = 0; j < length; j++){
			num = rand() % 100;
			if (debug) num = 1;
			h_A[j + i * length] = num;
			h_A_s[j + i * length] = num;
			num = rand() % 100;
			if (debug) num = 1;
			h_B[j + i * length] = num;
			h_B_s[j + i * length] = num;
			h_C[j + i * length] = 0;
		}
	}

	// Check memory allocation for errors
	cudaCheckErrors();

	// Print initial matrices A and B
	// if debug, then should be filled with 1s
	if (debug || printouts) {
		printf("Matrix A:\n");
		print_matrix(h_A, length);
		printf("Matrix B:\n");
		print_matrix(h_B, length);
	}

	// Define block/grid dimensions and launch stencil_2d kernel
	int gridSize = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	stencil_2d<<<grid,block>>>(h_A + RADIUS * length + RADIUS, h_A_s + RADIUS * length + RADIUS, length);
	stencil_2d<<<grid,block>>>(h_B + RADIUS * length + RADIUS, h_B_s + RADIUS * length + RADIUS, length);
	cudaDeviceSynchronize();

	// Print matrices after stencil_2d, A_s and B_s
	// if debug, then RADIUS wide halo of 1s with (4 * RADIUS + 1) center (13 for DSIZE = 512, RADIUS = 3)
	if (debug || printouts) {
		printf("Stenciled matrix A:\n");
		print_matrix(h_A_s, length);
		check_stencil_2d(h_A, h_A_s, length);
		printf("Stenciled matrix B:\n");
		print_matrix(h_B_s, length);
		check_stencil_2d(h_B, h_B_s, length);
	}

	// Define block/grid dimensions and launch dot_product kernel
    dim3 grid2(length, length);
	dim3 block2(BLOCK_SIZE, BLOCK_SIZE);
	dot_product<<<grid2, block2>>>(h_A_s, h_B_s, h_C, length);
	cudaDeviceSynchronize();

	// Print matrix after dot product, C
	// if debug, then (RADIUS * RADIUS) corners should be (DSIZE + 2 * RADIUS) (518 for DSIZE = 512, RADIUS = 3), 
	// RADIUS wide edges should be (DSIZE * (4 * RADIUS + 1) + 2 * RADIUS) (6662), 
	// center should be (DSIZE * (4 * RADIUS + 1) * (4 * RADIUS + 1) + 2 * RADIUS) (86534)
	if (debug || printouts){
		printf("Dot product matrix C:\n");
		print_matrix(h_C, length);
		check_dot_product(h_A_s, h_B_s, h_C, length);
	}

	// Free allocated memory
	cudaFree(h_A);
    cudaFree(h_B);
	cudaFree(h_A_s);
	cudaFree(h_B_s);
	cudaFree(h_C);
		
	printf("Finished!\n");
	return 0;
}