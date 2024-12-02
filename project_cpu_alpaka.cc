#include <alpaka/alpaka.hpp>
#include "config.hpp"
#include "workdiv.hpp"

constexpr std::size_t DSIZE = 512;
#define RADIUS 3

bool debug = true;
bool printouts = true;

struct stencil_2d {
	template <typename TAcc, typename T>
	ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ in, T* __restrict__ out) const {
		auto globalThreadId = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
		auto globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
		auto threadElementId = alpaka::mapIdx<1u>(globalThreadId, globalThreadExtent);

		int length = DSIZE + 2 * RADIUS;
		auto id = threadElementId[0];
		auto idx = id % length;
		auto idy = id / length;

		// Check if outside boundary
		if (idx < RADIUS || idx >= length - RADIUS|| idy < RADIUS || idy >= length - RADIUS) {
			out[idy * length + idx] = in[idy * length + idx];
			return;
		}

		// Apply the stencil
		int result = 0;
		for (int offset = -RADIUS; offset <= RADIUS; offset++){
			result += in[idy * length + idx + offset];
			if (offset != 0) {
				result += in[(idy + offset) * length + idx];
			}
		}
		out[idy * length + idx] = result;
	}
};


struct dot_product {
	template <typename TAcc, typename T>
	ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ A, T const* __restrict__ B, T* __restrict__ C) const {
		auto globalThreadId = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
		auto globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
		auto threadElementId = alpaka::mapIdx<1u>(globalThreadId, globalThreadExtent);

		int length = DSIZE + 2 * RADIUS;
		auto id = threadElementId[0];
		auto idx = id % length;
		auto idy = id / length;

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
};


void print_matrix(alpaka::BufCpu<int, Dim1D, uint32_t> matrix, int length) {
	int maxPrintOut = 10;
	for (int i = 0; i < maxPrintOut; i++) {
		for (int j = 0; j < maxPrintOut; j++) {
			printf("%d ", matrix[j + i * length]);
		}
		printf("\n");
	}
}


void check_stencil_2d(alpaka::BufCpu<int, Dim1D, uint32_t> in, alpaka::BufCpu<int, Dim1D, uint32_t> out, const int length) {
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


void check_dot_product(alpaka::BufCpu<int, Dim1D, uint32_t> A, alpaka::BufCpu<int, Dim1D, uint32_t> B, alpaka::BufCpu<int, Dim1D, uint32_t> C, int length) {	
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
	// initialise the accelerator platform
 	Platform platform;

	// require at least one device
	std::size_t n = alpaka::getDevCount(platform);
	if (n == 0) {
		exit(EXIT_FAILURE);
	}

	// use the single host device
	HostPlatform host_platform;
	Host host = alpaka::getDevByIdx(host_platform, 0u);
	std::cout << "Host:   " << alpaka::getName(host) << '\n';

	std::cout << "Create pointers and allocate matrices\n";
    // Create pointers and allocate matrices
    uint32_t length = (DSIZE + 2 * RADIUS);
	auto A = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{length * length});
	auto B = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{length * length});
	auto A_s = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{length * length});
	auto B_s = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{length * length});
	auto C = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{length * length});

	std::cout << "Fill in matrices\n";
	// Fill in matrices
	int num = 0;
	for (auto i = 0; i < length; i++){
		for (auto j = 0; j < length; j++){
			num = rand() % 100;
			if (debug) num = 1;
			A[i * length + j] = num;
			A_s[i * length + j] = num;
			num = rand() % 100;
			if (debug) num = 1;
			B[i * length + j] = num;
			B_s[i * length + j] = num;
			C[i * length + j] = 0;
		}
	}

	std::cout << "Use the first device\n";
	// Use the first device
	Device device = alpaka::getDevByIdx(platform, 0u);
	std::cout << "Device: " << alpaka::getName(device) << '\n';

	// Create a work queue
	Queue queue(device);

	// Allocate device memory 
	auto d_A = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{length * length});
	auto d_B = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{length * length});
	auto d_A_s = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{length * length});
	auto d_B_s = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{length * length});
	auto d_C = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{length * length});

	// Print initial matrices A and B
	// if debug, then should be filled with 1s
	if (debug || printouts) {
		printf("Matrix A:\n");
		print_matrix(A, length);
		printf("Matrix B:\n");
		print_matrix(B, length);
	}

	// Copy to device
	alpaka::memcpy(queue, d_A, A);
	alpaka::memcpy(queue, d_B, B);

	alpaka::memset(queue, d_A_s, 0);
	alpaka::memset(queue, d_B_s, 0);
	alpaka::memset(queue, d_C, 0);

	// Launch stencil_2d kernel
	auto div = make_workdiv<Acc1D>(1024, 1024);
    std::cout << "Testing VectorAddKernel with scalar indices with a grid of " << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x " << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x " << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements" << std::endl;
	alpaka::exec<Acc1D>(queue, div, stencil_2d{}, d_A.data(), d_A_s.data());
	alpaka::exec<Acc1D>(queue, div, stencil_2d{}, d_B.data(), d_B_s.data());

	// Copy results back to host
	alpaka::memcpy(queue, A_s, d_A_s);
	alpaka::memcpy(queue, B_s, d_B_s);
	alpaka::wait(queue);

	// Print matrices after stencil_2d, A_s and B_s
	// if debug, then RADIUS wide halo of 1s with (4 * RADIUS + 1) center (13 for DSIZE = 512, RADIUS = 3)
	if (debug || printouts) {
		printf("Stenciled matrix A:\n");
		print_matrix(A_s, length);
		check_stencil_2d(A, A_s, length);
		printf("Stenciled matrix B:\n");
		print_matrix(B_s, length);
		check_stencil_2d(B, B_s, length);
	}

	// Launch dot_product kernel
	alpaka::exec<Acc1D>(queue, div, dot_product{}, d_A_s.data(), d_B_s.data(), d_C.data());

	// Copy results back to host
    alpaka::memcpy(queue, C, d_C);
    alpaka::wait(queue);

	// Print matrix after dot product, C
	// if debug, then (RADIUS * RADIUS) corners should be (DSIZE + 2 * RADIUS) (518 for DSIZE = 512, RADIUS = 3), 
	// RADIUS wide edges should be (DSIZE * (4 * RADIUS + 1) + 2 * RADIUS) (6662), 
	// center should be (DSIZE * (4 * RADIUS + 1) * (4 * RADIUS + 1) + 2 * RADIUS) (86534)
	if (debug || printouts){
		printf("Dot product matrix C:\n");
		print_matrix(C, length);
		check_dot_product(A_s, B_s, C, length);
	}

	printf("Finished!\n");
	return 0;
}