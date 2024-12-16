#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>

#define NX 32
#define NY 32
#define ITER 3000

// Directions for lattice Boltzmann
enum Direction { C = 0, E, S, W, N, NE, SE, SW, NW };

// Constants
constexpr float RHO_0 = 1.0f;
constexpr float U_0 = 0.1f;
constexpr float RE = 1000.0f;
constexpr float VISCOSITY = (NY - 1) * U_0 / RE;
constexpr float TAU = (6 * VISCOSITY + 1) / 2;

// CUDA error handling
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(x) << " at " << __FILE__ << ":" << __LINE__; \
    exit(EXIT_FAILURE);}} while(0)

// Kernel for initialization
__global__ void initialize(float* f, float* feq, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = nx * ny;

    if (idx < total_cells) {
        float rho = RHO_0;
        float ux = 0.0f, uy = 0.0f;
        float usqr = ux * ux + uy * uy;

        f[idx * 9 + C] = (4.0f / 9.0f) * rho;
        f[idx * 9 + E] = f[idx * 9 + W] = f[idx * 9 + N] = f[idx * 9 + S] = (1.0f / 9.0f) * rho;
        f[idx * 9 + NE] = f[idx * 9 + NW] = f[idx * 9 + SE] = f[idx * 9 + SW] = (1.0f / 36.0f) * rho;

        feq[idx * 9 + C] = f[idx * 9 + C];
        feq[idx * 9 + E] = f[idx * 9 + E];
        feq[idx * 9 + S] = f[idx * 9 + S];
        feq[idx * 9 + W] = f[idx * 9 + W];
        feq[idx * 9 + N] = f[idx * 9 + N];
        feq[idx * 9 + NE] = f[idx * 9 + NE];
        feq[idx * 9 + SE] = f[idx * 9 + SE];
        feq[idx * 9 + SW] = f[idx * 9 + SW];
        feq[idx * 9 + NW] = f[idx * 9 + NW];
    }
}

__global__ void find_values(int *FL,int *WALL,int *DR, int **mesh, int nx, int ny, int *counterFL,int *counterWALL,int *counterDR){
	int x = INDEX / ny;
	int y = INDEX % nx;
	if(mesh[x][y]==0){
		int pos = atomicAdd(counterFL, 1); // Atomic increment to get unique position
		FL[pos] = x * ny + y; // Store the flattened index
	}else if(mesh[x][y]==1){
		int pos = atomicAdd(counterWALL, 1); // Atomic increment to get unique position
		WALL[pos] = x * ny + y; // Store the flattened index
	}else if (mesh[x][y]==2){
		int pos = atomicAdd(counterDR, 1); // Atomic increment to get unique position
		DR[pos] = x * ny + y; // Store the flattened index
	}
}

// Kernel for collision step
__global__ void collision(float* f, float* feq, float* rho, float* ux, float* uy, int* fluid_cells, int num_fluid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_fluid) {
        int cell_idx = fluid_cells[idx];
        float local_rho = 0.0f;
        float local_ux = 0.0f, local_uy = 0.0f;

        for (int d = 0; d < 9; d++) {
            local_rho += f[cell_idx * 9 + d];
        }
        for (int d = 0; d < 9; d++) {
            local_ux += (f[cell_idx * 9 + E] - f[cell_idx * 9 + W] +
                         f[cell_idx * 9 + NE] - f[cell_idx * 9 + NW]) / local_rho;
            local_uy += (f[cell_idx * 9 + N] - f[cell_idx * 9 + S] +
                         f[cell_idx * 9 + NE] - f[cell_idx * 9 + SW]) / local_rho;
        }

        rho[cell_idx] = local_rho;
        ux[cell_idx] = local_ux;
        uy[cell_idx] = local_uy;

        // Compute equilibrium distribution function
        float usqr = local_ux * local_ux + local_uy * local_uy;
        feq[cell_idx * 9 + C] = (4.0f / 9.0f) * local_rho * (1.0f - 1.5f * usqr);
        feq[cell_idx * 9 + E] = (1.0f / 9.0f) * local_rho * (1.0f + 3.0f * local_ux + 4.5f * local_ux * local_ux - 1.5f * usqr);
        // Continue for all other directions...
    }
}

// Main host code
int main() {
    // Grid and block sizes
    dim3 threadsPerBlock(256);
    dim3 numBlocks((NX * NY + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Allocate and initialize device memory
    float* d_f, *d_feq, *d_rho, *d_ux, *d_uy;
    CUDA_CALL(cudaMalloc(&d_f, NX * NY * 9 * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_feq, NX * NY * 9 * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_rho, NX * NY * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_ux, NX * NY * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_uy, NX * NY * sizeof(float)));

    initialize<<<numBlocks, threadsPerBlock>>>(d_f, d_feq, NX, NY);

    // Implement collision, propagation, and boundary conditions in a loop

    // Free memory
    CUDA_CALL(cudaFree(d_f));
    CUDA_CALL(cudaFree(d_feq));
    CUDA_CALL(cudaFree(d_rho));
    CUDA_CALL(cudaFree(d_ux));
    CUDA_CALL(cudaFree(d_uy));

    return 0;
}
