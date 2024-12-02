#include <stdio.h>
#include <cstdlib>

#define DSIZE 512
#define RADIUS 3

bool debug = false;
bool printouts = true;

void stencil_2d(const int *in, int *out, int length) {
	// Apply the stencil
	int result;
    for (int i = RADIUS; i < length - RADIUS; i++) {
        for (int j = RADIUS; j < length - RADIUS; j++) {
			result = 0;
            for (int offset = -RADIUS; offset <= RADIUS; offset++) {
		        result += in[j + (i + offset) * length];
		        if (offset != 0) {
			        result += in[(j + offset) + i * length];
		        }
            }
            out[j + i * length] = result;
        }
    }
}


void dot_product(const int *A, const int *B, int *C, int length) {	
    // Apply dot product
	int result;
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            result = 0;
            for (int k = 0; k < length; k++){
                result += A[k + i * length] * B[j + k * length];
            }
            C[j + i * length] = result;
        }
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


int main() {
	
    // Create pointers and allocate matrices
    int length = (DSIZE + 2 * RADIUS);
	int *A = (int *)malloc(length * length * sizeof(int));
    int *B = (int *)malloc(length * length * sizeof(int));
    int *A_s = (int *)malloc(length * length * sizeof(int));
    int *B_s = (int *)malloc(length * length * sizeof(int));
    int *C = (int *)malloc(length * length * sizeof(int));

	// Fill in matrices
	int num = 0;
	for (int i = 0; i < length; i++){
		for (int j = 0; j < length; j++){
			num = rand() % 100;
			if (debug) num = 1;
			A[i * length +j] = num;
			A_s[i * length +j] = num;
			num = rand() % 100;
			if (debug) num = 1;
			B[i * length +j] = num;
			B_s[i * length +j] = num;
			C[i * length +j] = 0;
		}
	}

	// Print initial matrices A and B
	// if debug, then should be filled with 1s
	if (debug || printouts) {
		printf("Matrix A:\n");
		print_matrix(A, length);
		printf("Matrix B:\n");
		print_matrix(B, length);
	}

	// Send to stencil_2d function
	stencil_2d(A, A_s, length);
	stencil_2d(B, B_s, length);

	// Print matrices after stencil_2d, A_s and B_s
	// if debug, then RADIUS wide halo of 1s with (4 * RADIUS + 1) center (13 for DSIZE = 512, RADIUS = 3)
	if (debug || printouts) {
		printf("Stenciled matrix A:\n");
		print_matrix(A_s, length);
		printf("Stenciled matrix B:\n");
		print_matrix(B_s, length);
	}

	//Send to dot_product function
	dot_product(A_s, B_s, C, length);

	// Print matrix after dot product, C
	// if debug, then (RADIUS * RADIUS) corners should be (DSIZE + 2 * RADIUS) (518 for DSIZE = 512, RADIUS = 3), 
	// RADIUS wide edges should be (DSIZE * (4 * RADIUS + 1) + 2 * RADIUS) (6662), 
	// center should be (DSIZE * (4 * RADIUS + 1) * (4 * RADIUS + 1) + 2 * RADIUS) (86534)
	if (debug || printouts){
		printf("Dot product matrix C:\n");
		print_matrix(C, length);
	}
	
    // Free memory
    free(A);
    free(B);
    free(A_s);
    free(B_s);
    free(C);

	printf("Finished!\n");
	return 0;
}