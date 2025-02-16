#include <assert.h> // Pour les assertions
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream> // Pour debug ou affichage

#include "args.hxx" // Pour parser les arguments

// initialisation des blocs
#define TILE_SIZE 16
#define BLOCK_SIZE 32


// initialisation des directions
#define C	0
#define E	1
#define S	2
#define W	3
#define N	4
#define NE	5
#define SE	6
#define SW	7
#define NW	8

#define BLOCK_INDEX ((blockDim.x * threadIdx.y) + threadIdx.x)
#define INDEX (gridDim.x * blockDim.x * (blockDim.y * blockIdx.y + threadIdx.y) + blockIdx.x * blockDim.x + threadIdx.x)
#define INDEX_X (blockIdx.x * blockDim.x + threadIdx.x)
#define INDEX_Y	(blockDim.y * blockIdx.y + threadIdx.y)
#define OFFSET_Y (gridDim.x * blockDim.x)
#define INDEX_FROM(x, y) ((OFFSET_Y) * (y) + (x))

int nx = 256;
int ny = 256;

typedef union Directions {
	float direction[9];
	struct {
		float c;
		float e;
		float s;
		float w;
		float n;
		float ne;
		float se;
		float sw;
		float nw;
	};
} directions_t;

void copy (directions_t *d1, directions_t *d2){
	for(int i = 0; i<9 ; i++){
		d1->direction[i] = d2->direction[i];
	}
}

template <typename T>
void print_matrix(T* table, int size) {
	if (size == 0) {
		return;
	} else if (size == 1) {
		std::cout << table << std::endl;
	} else if (size == nx*ny) {
		for (int i = 0; i < size; i++) {
			std::cout << table[i] << " ";
			if ((i+1) % nx == 0) {
				std::cout << std::endl;
			}
		}
		std::cout << std::endl;
	} else {
		for (int i = 0; i < size; i++) {
			std::cout << table[i] << " ";
		}
		std::cout << std::endl;
	}
}

void printdirection (directions_t *f, int nx, int ny){
	for(int i = 0; i<nx; i++){
		for(int j = 0; j<ny; j++){
			std::cout << "[" << i << "][" << j << "]" << std::endl;
			std::cout << f[i*ny+j].nw << " ";
			std::cout << f[i*ny+j].n << " ";
			std::cout << f[i*ny+j].ne << std::endl;
			std::cout << f[i*ny+j].w << " ";
			std::cout << f[i*ny+j].c << " ";
			std::cout << f[i*ny+j].e << std::endl;
			std::cout << f[i*ny+j].sw << " ";
			std::cout << f[i*ny+j].s << " ";
			std::cout << f[i*ny+j].se << std::endl;
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

// fonction pour afficher toutes les données (rho, ux, uy, f, feq...) passées en paramètre
void printData(int nx, int ny, int iter, int Re, float rho_0, float u_0, float viscosity, float tau, int* mesh, directions_t *f, directions_t *feq, float *rho, float *ux, float *uy, float *usqr, bool *DR, bool *WL, bool *FL) {
	std::cout << "##########################################################################################################\n##########################################################################################################\n##########################################################################################################\n##########################################################################################################" << std::endl;
	std::cout << "nx = " << nx << std::endl;
	std::cout << "ny = " << ny << std::endl;
	std::cout << "iter = " << iter << std::endl;
	std::cout << "Re = " << Re << std::endl;
	std::cout << "rho_0 = " << rho_0 << std::endl;
	std::cout << "u_0 = " << u_0 << std::endl;
	std::cout << "viscosity = " << viscosity << std::endl;
	std::cout << "tau_0 = " << tau << std::endl;
	std::cout << std::endl;

	std::cout << "Affichage de  : mesh" << std::endl;
	print_matrix(mesh, nx*ny);

	std::cout << "Affichage de  : f" << std::endl;
	printdirection(f, nx, ny);

	std::cout << "Affichage de  : feq" << std::endl;
	printdirection(feq, nx, ny);

	std::cout << "Affichage de  : rho" << std::endl;
	print_matrix(rho, nx*ny);

	std::cout << "Affichage de  : ux" << std::endl;
	print_matrix(ux, nx*ny);

	std::cout << "Affichage de  : uy" << std::endl;
	print_matrix(uy, nx*ny);

	std::cout << "Affichage de  : usqr" << std::endl;
	print_matrix(usqr, nx*ny);

	std::cout << "Affichage de  : DR" << std::endl;
	print_matrix(DR, nx*ny);

	std::cout << "Affichage de  : WALL" << std::endl;
	print_matrix(WL, nx*ny);

	std::cout << "Affichage de  : FL" << std::endl;
	print_matrix(FL, nx*ny);
}

__global__ void collision_step_shared (
	directions_t *f,
	directions_t *feq,
	float *rho,
	float *ux,
	float *uy,
	float *usqr,
	bool *DR,
	bool *WL,
	bool *FL,
	float u_0,
	float tau,
	int nf) {
	
	// Shared memory initialization
	extern __shared__ directions_t ffeq_buffer[];
	directions_t* f_buffer = ffeq_buffer;
	directions_t* feq_buffer = &ffeq_buffer[nf];

	// Shared memory filling from global memory (no need to copy feq)
	f_buffer[BLOCK_INDEX] = f[INDEX];

	// Sync
	__syncthreads();

	// Macroscopic density
	rho[INDEX] = 0;
	for (int i=0; i<9; i++) {
		rho[INDEX] += f_buffer[BLOCK_INDEX].direction[i];
	}

	// Macroscopic velocities
	ux[INDEX] = (DR[INDEX] ? u_0 : (f_buffer[BLOCK_INDEX].e - f_buffer[BLOCK_INDEX].w + f_buffer[BLOCK_INDEX].ne + f_buffer[BLOCK_INDEX].se - f_buffer[BLOCK_INDEX].sw - f_buffer[BLOCK_INDEX].nw) / rho[INDEX]);
	uy[INDEX] = (DR[INDEX] ? 0 : (f_buffer[BLOCK_INDEX].n - f_buffer[BLOCK_INDEX].s + f_buffer[BLOCK_INDEX].ne + f_buffer[BLOCK_INDEX].nw - f_buffer[BLOCK_INDEX].se - f_buffer[BLOCK_INDEX].sw) / rho[INDEX]);
	usqr[INDEX] = ux[INDEX] * ux[INDEX] + uy[INDEX] * uy[INDEX];

	feq_buffer[BLOCK_INDEX].c = (4./9.) * rho[INDEX] * (1. - 1.5 * usqr[INDEX]);
	feq_buffer[BLOCK_INDEX].e = (1./9.) * rho[INDEX] * (1. + 3. * ux[INDEX] + 4.5 * (ux[INDEX] * ux[INDEX]) - 1.5 * usqr[INDEX]);
	feq_buffer[BLOCK_INDEX].s = (1./9.) * rho[INDEX] * (1. - 3. * uy[INDEX] + 4.5 * (uy[INDEX] * uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq_buffer[BLOCK_INDEX].w = (1./9.) * rho[INDEX] * (1. - 3. * ux[INDEX] + 4.5 * (ux[INDEX] * ux[INDEX]) - 1.5 * usqr[INDEX]);
	feq_buffer[BLOCK_INDEX].n = (1./9.) * rho[INDEX] * (1. + 3. * uy[INDEX] + 4.5 * (uy[INDEX] * uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq_buffer[BLOCK_INDEX].ne = (1./36.) * rho[INDEX] * (1. + 3. * (ux[INDEX] + uy[INDEX]) + 4.5 * (ux[INDEX] + uy[INDEX]) * (ux[INDEX] + uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq_buffer[BLOCK_INDEX].se = (1./36.) * rho[INDEX] * (1. + 3. * (ux[INDEX] - uy[INDEX]) + 4.5 * (ux[INDEX] - uy[INDEX]) * (ux[INDEX] - uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq_buffer[BLOCK_INDEX].sw = (1./36.) * rho[INDEX] * (1. + 3. * (-ux[INDEX] - uy[INDEX]) + 4.5 * (-ux[INDEX] - uy[INDEX]) * (-ux[INDEX] - uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq_buffer[BLOCK_INDEX].nw = (1./36.) * rho[INDEX] * (1. + 3. * (-ux[INDEX] + uy[INDEX]) + 4.5 * (-ux[INDEX] + uy[INDEX]) * (-ux[INDEX] + uy[INDEX]) - 1.5 * usqr[INDEX]);

	if(WL[INDEX]) {
		f_buffer[BLOCK_INDEX].e = f_buffer[BLOCK_INDEX].w;
		f_buffer[BLOCK_INDEX].s = f_buffer[BLOCK_INDEX].n;
		f_buffer[BLOCK_INDEX].w = f_buffer[BLOCK_INDEX].e;
		f_buffer[BLOCK_INDEX].n = f_buffer[BLOCK_INDEX].s;
		f_buffer[BLOCK_INDEX].ne = f_buffer[BLOCK_INDEX].sw;
		f_buffer[BLOCK_INDEX].se = f_buffer[BLOCK_INDEX].nw;
		f_buffer[BLOCK_INDEX].sw = f_buffer[BLOCK_INDEX].ne;
		f_buffer[BLOCK_INDEX].nw = f_buffer[BLOCK_INDEX].se;
	} else if (DR[INDEX]) {
		for (int i=0; i<9; i++) {
			f_buffer[BLOCK_INDEX].direction[i] = feq_buffer[BLOCK_INDEX].direction[i];
		}
	} else {
		for (int i=0; i<9; i++) {
			f_buffer[BLOCK_INDEX].direction[i] = f_buffer[BLOCK_INDEX].direction[i] * (1. - 1. / tau) + feq_buffer[BLOCK_INDEX].direction[i] / tau;
		}
	}

	__syncthreads();

	// Writing back to global memory
	f[INDEX] = f_buffer[BLOCK_INDEX];
	feq[INDEX] = feq_buffer[BLOCK_INDEX];
}

__global__ void collision_step (
	directions_t *f,
	directions_t *feq,
	float *rho,
	float *ux,
	float *uy,
	float *usqr,
	bool *DR,
	bool *WL,
	bool *FL,
	float u_0,
	float tau) {
	
	// Macroscopic density
	rho[INDEX] = 0;
	for (int i=0; i<9; i++) {
		rho[INDEX] += f[INDEX].direction[i];
	}

	// Macroscopic velocities
	ux[INDEX] = (DR[INDEX] ? u_0 : (f[INDEX].e - f[INDEX].w + f[INDEX].ne + f[INDEX].se - f[INDEX].sw - f[INDEX].nw) / rho[INDEX]);
	uy[INDEX] = (DR[INDEX] ? 0 : (f[INDEX].n - f[INDEX].s + f[INDEX].ne + f[INDEX].nw - f[INDEX].se - f[INDEX].sw) / rho[INDEX]);
	usqr[INDEX] = ux[INDEX] * ux[INDEX] + uy[INDEX] * uy[INDEX];

	feq[INDEX].c = (4./9.) * rho[INDEX] * (1. - 1.5 * usqr[INDEX]);
	feq[INDEX].e = (1./9.) * rho[INDEX] * (1. + 3. * ux[INDEX] + 4.5 * (ux[INDEX] * ux[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].s = (1./9.) * rho[INDEX] * (1. - 3. * uy[INDEX] + 4.5 * (uy[INDEX] * uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].w = (1./9.) * rho[INDEX] * (1. - 3. * ux[INDEX] + 4.5 * (ux[INDEX] * ux[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].n = (1./9.) * rho[INDEX] * (1. + 3. * uy[INDEX] + 4.5 * (uy[INDEX] * uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].ne = (1./36.) * rho[INDEX] * (1. + 3. * (ux[INDEX] + uy[INDEX]) + 4.5 * (ux[INDEX] + uy[INDEX]) * (ux[INDEX] + uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].se = (1./36.) * rho[INDEX] * (1. + 3. * (ux[INDEX] - uy[INDEX]) + 4.5 * (ux[INDEX] - uy[INDEX]) * (ux[INDEX] - uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].sw = (1./36.) * rho[INDEX] * (1. + 3. * (-ux[INDEX] - uy[INDEX]) + 4.5 * (-ux[INDEX] - uy[INDEX]) * (-ux[INDEX] - uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].nw = (1./36.) * rho[INDEX] * (1. + 3. * (-ux[INDEX] + uy[INDEX]) + 4.5 * (-ux[INDEX] + uy[INDEX]) * (-ux[INDEX] + uy[INDEX]) - 1.5 * usqr[INDEX]);

	if(WL[INDEX]) {
		f[INDEX].e = f[INDEX].w;
		f[INDEX].s = f[INDEX].n;
		f[INDEX].w = f[INDEX].e;
		f[INDEX].n = f[INDEX].s;
		f[INDEX].ne = f[INDEX].sw;
		f[INDEX].se = f[INDEX].nw;
		f[INDEX].sw = f[INDEX].ne;
		f[INDEX].nw = f[INDEX].se;
	} else if (DR[INDEX]) {
		for (int i=0; i<9; i++) {
			f[INDEX].direction[i] = feq[INDEX].direction[i];
		}
	} else {
		for (int i=0; i<9; i++) {
			f[INDEX].direction[i] = f[INDEX].direction[i] * (1. - 1. / tau) + feq[INDEX].direction[i] / tau;
		}
	}
}

__global__ void propagation_step_shared(directions_t *f_src, directions_t *f_dst, int nx, int ny) {
	extern __shared__ directions_t buffer[];

	if(INDEX_X < nx-1)	buffer[INDEX_FROM(INDEX_X + 1, INDEX_Y)].e = f_src[INDEX].e;
	if(INDEX_X > 0)		buffer[INDEX_FROM(INDEX_X - 1, INDEX_Y)].w = f_src[INDEX].w;
	if(INDEX_Y < ny-1)	buffer[INDEX_FROM(INDEX_X, INDEX_Y + 1)].s = f_src[INDEX].s;
	if(INDEX_Y > 0)		buffer[INDEX_FROM(INDEX_X, INDEX_Y - 1)].n = f_src[INDEX].n;
	if(INDEX_X < nx-1 && INDEX_Y < ny-1)	buffer[INDEX_FROM(INDEX_X + 1, INDEX_Y + 1)].se = f_src[INDEX].se;
	if(INDEX_X < nx-1 && INDEX_Y > 0)		buffer[INDEX_FROM(INDEX_X + 1, INDEX_Y - 1)].ne = f_src[INDEX].ne;
	if(INDEX_X > 0 && INDEX_Y < ny-1)		buffer[INDEX_FROM(INDEX_X - 1, INDEX_Y + 1)].sw = f_src[INDEX].sw;
	if(INDEX_X > 0 && INDEX_Y > 0)			buffer[INDEX_FROM(INDEX_X - 1, INDEX_Y - 1)].nw = f_src[INDEX].nw;

	__syncthreads();

	f_dst[INDEX] = buffer[INDEX];

	// Maybe useless
	__syncthreads();
}

__global__ void propagation_step(directions_t *f_src, directions_t *f_dst, int nx, int ny) {
	if(INDEX_X < nx-1)	f_dst[INDEX_FROM(INDEX_X + 1, INDEX_Y)].e = f_src[INDEX].e;
	if(INDEX_X > 0)		f_dst[INDEX_FROM(INDEX_X - 1, INDEX_Y)].w = f_src[INDEX].w;
	if(INDEX_Y < ny-1)	f_dst[INDEX_FROM(INDEX_X, INDEX_Y + 1)].s = f_src[INDEX].s;
	if(INDEX_Y > 0)		f_dst[INDEX_FROM(INDEX_X, INDEX_Y - 1)].n = f_src[INDEX].n;
	if(INDEX_X < nx-1 && INDEX_Y < ny-1)	f_dst[INDEX_FROM(INDEX_X + 1, INDEX_Y + 1)].se = f_src[INDEX].se;
	if(INDEX_X < nx-1 && INDEX_Y > 0)		f_dst[INDEX_FROM(INDEX_X + 1, INDEX_Y - 1)].ne = f_src[INDEX].ne;
	if(INDEX_X > 0 && INDEX_Y < ny-1)		f_dst[INDEX_FROM(INDEX_X - 1, INDEX_Y + 1)].sw = f_src[INDEX].sw;
	if(INDEX_X > 0 && INDEX_Y > 0)			f_dst[INDEX_FROM(INDEX_X - 1, INDEX_Y - 1)].nw = f_src[INDEX].nw;

}

__global__ void index_testing(directions_t *f, bool *WL) {
	for(int i=0; i<9; i++) {
		if (WL[INDEX]) {
			f[INDEX].direction[i] = INDEX;
		} else {
			f[INDEX].direction[i] = 0;
		}
	}
}

__global__ void init_mesh(int *mesh, int nx, int ny){

	if(INDEX < ny)
		mesh[INDEX]=2; // le fluide est injecté à gauche
	else if (INDEX % ny == 0 || INDEX % ny == ny-1 || INDEX > nx*ny-ny)
		mesh[INDEX] = 1; // les bords sont des murs
	else
		mesh[INDEX] = 0; // le reste est du fluide
    
}

__global__ void init_sim(
	directions_t *f,
	directions_t *feq,
	float *rho,
	float *ux,
	float *uy,
	float *usqr,
	float rho_0
	){

	for(int j = 0; j<9; j++) {
		feq[INDEX].direction[j] = 0;
	}

	f[INDEX].c = rho_0 * 4. / 9.;
	f[INDEX].e = rho_0 / 9.;
	f[INDEX].s = rho_0 / 9.;
	f[INDEX].w = rho_0 / 9.;
	f[INDEX].n = rho_0 / 9.;
	f[INDEX].ne = rho_0 / 36.;
	f[INDEX].se = rho_0 / 36.;
	f[INDEX].sw = rho_0 / 36.;
	f[INDEX].nw = rho_0 / 36.;

	rho[INDEX]=0; 
	ux[INDEX]=0; 
	uy[INDEX]=0; 
	usqr[INDEX]=0;
    
}

__global__ void init_fluid(
	int *mesh,
	bool *DR,
	bool *WL,
	bool *FL){
	
	if(mesh[INDEX]==0){
		FL[INDEX] = true;
		WL[INDEX] = false; // Store the flattened index
		DR[INDEX] = false; // Store the flattened index
		
	}else if(mesh[INDEX]==1){
		FL[INDEX] = false;
		WL[INDEX] = true; // Store the flattened index
		DR[INDEX] = false; // Store the flattened index
	}else if (mesh[INDEX]==2){
		FL[INDEX] = false;
		WL[INDEX] = false; // Store the flattened index
		DR[INDEX] = true; // Store the flattened index
	}


}

int main (int argc, char** argv){

	int* matrix = NULL;

	// Define parser
	args::ArgumentParser parser("main", "This is a main program for the LBM simulation");

	// Set parser value
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::Flag print(parser, "print", "print the matrix at the end", {"p"});
	args::ValueFlag<std::string> importfile(parser, "file", "File to read", {"f"});
	args::ValueFlag<int> width(parser, "width", "Width of matrix ", {"w"}, 32);
	args::ValueFlag<int> height(parser, "height", "Height of matrix ", {"h"},32);
	args::ValueFlag<int> iterations(parser, "iteration", "Number of iterations", {"i"}, 3000);
	args::Flag is_shared(parser, "shared", "Use shared memory", {"s"});

	// Invoke parser
	try {
		parser.ParseCLI(argc, argv);
	} catch (args::Help) {
		std::cout << parser;
		return 0;
	} catch (args::ParseError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	} catch (args::ValidationError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

    // initialisation des parametres de la simulation
    int iter, Re;
    if (width && height) { //initialisation de la taille de la grille
		nx = args::get(width);
		ny = args::get(height);
	}
	if (iterations) //initialisation du nombre d'iterations
		iter = args::get(iterations);
	else
		iter = 60000;
	Re=1000; // nombre de Reynolds

    // initialisation des variables
    float  rho_0, u_0, viscosity, tau;
    rho_0 = 1; // densite initiale
    u_0 = 0.1; // vitesse initiale
	viscosity = (ny-1)*u_0/Re; // viscosite
    tau = (6*viscosity+1)/2; // relaxation time
	

    // initialisation de la grille de la simulation
    int* mesh = new int[nx*ny]; // 0 = fluid, 1 = wall, 2 = driving fluid
    
    directions_t *f, *feq;
	float *rho, *ux, *uy, *usqr;
	f = new directions_t[nx*ny]; // distribution function
	feq = new directions_t[nx*ny]; // equilibrium distribution function
	rho = new float[nx*ny]; // macroscopic density
	ux = new float[nx*ny]; // macroscopic velocity in direction x
	uy = new float[nx*ny]; // macroscopic velocity in direction y
	usqr = new float[nx*ny]; // helper variable


	bool *DR, *WALL, *FL;
	DR = new bool[nx*ny]; // driving fluid
	WALL = new bool[nx*ny]; // wall
	FL = new bool[nx*ny]; // fluid
	
	std::string path = args::get(importfile);
	if (!path.empty()) {
		FILE *file = fopen(path.c_str(), "r");
		if (file == NULL) {
			perror("Erreur lors de l'ouverture du fichier");
			return EXIT_FAILURE;
		}

		if (fscanf(file, "%d %d", &nx, &ny) != 2) {
			fprintf(stderr, "Erreur de lecture des dimensions de la matrice\n");
			fclose(file);
			return EXIT_FAILURE;
		}

		matrix = (int *)malloc(ny * nx * sizeof(int));
		if (matrix == NULL) {
			fprintf(stderr, "Erreur d'allocation mémoire\n");
			fclose(file);
			return EXIT_FAILURE;
		}

		for (int i = 0; i < ny*nx; i++) {
			if (fscanf(file, "%d", &matrix[i]) != 1) {
				fprintf(stderr, "Erreur de lecture des valeurs\n");
				return EXIT_FAILURE;
			}
		}

		fclose(file);

	}
	
	//================= CUDA =================
	directions_t *d_f, *d_fswap, *d_feq, *d_ftmp;
	float *d_rho, *d_ux, *d_uy, *d_usqr;
	bool *d_DR, *d_WALL, *d_FL;
	int *d_mesh;
	

	cudaMalloc(&d_f, nx*ny*sizeof(directions_t));
	cudaMalloc(&d_fswap, nx*ny*sizeof(directions_t));
	cudaMalloc(&d_feq, nx*ny*sizeof(directions_t));
	cudaMalloc(&d_rho, nx*ny*sizeof(float));
	cudaMalloc(&d_ux, nx*ny*sizeof(float));
	cudaMalloc(&d_uy, nx*ny*sizeof(float));
	cudaMalloc(&d_usqr, nx*ny*sizeof(float));
	cudaMalloc(&d_DR, nx*ny*sizeof(bool));
	cudaMalloc(&d_WALL, nx*ny*sizeof(bool));
	cudaMalloc(&d_FL, nx*ny*sizeof(bool));
	cudaMalloc(&d_mesh, nx*ny*sizeof(int));
	
	cudaEvent_t start,stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);

	//========== INITIALISATION ==========

	dim3 threads;
	dim3 grid;

	if(is_shared) {
		printf("Using shared memory\n");
		threads = dim3(TILE_SIZE, TILE_SIZE);
		grid = dim3(nx / TILE_SIZE, ny / TILE_SIZE);
	} else {
		threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
		grid = dim3(nx / BLOCK_SIZE, ny / BLOCK_SIZE);
	}

	if(path.empty()) {
		init_mesh<<<grid, threads>>>(
			d_mesh,
			nx,
			ny
		);
	} else {
		cudaMemcpy(d_mesh, matrix, nx*ny*sizeof(int), cudaMemcpyHostToDevice);
	}


	if (print) {
		cudaMemcpy(mesh, d_mesh, nx*ny*sizeof(int), cudaMemcpyDeviceToHost);
		printData(nx, ny, iter, Re, rho_0, u_0, viscosity, tau, mesh, f, feq, rho, ux, uy, usqr, DR, WALL, FL);
	}

	init_sim<<<grid, threads>>>(
		d_f,
		d_feq,
		d_rho,
		d_ux,
		d_uy,
		d_usqr,
		rho_0
	);
	
	
	init_fluid<<<grid, threads>>>(
		d_mesh,
		d_DR,
		d_WALL,
		d_FL
	);
	
	cudaMemcpy(d_fswap, d_f, nx*ny*sizeof(directions_t), cudaMemcpyDeviceToDevice);
	//============ MAIN LOOP =============
	
	for(int i=0; i<iter; i++) {
		if (is_shared) {
			collision_step_shared<<<grid, threads, TILE_SIZE * TILE_SIZE * sizeof(directions_t) * 2>>>(
				d_f,
				d_feq,
				d_rho,
				d_ux,
				d_uy,
				d_usqr,
				d_DR,
				d_WALL,
				d_FL,
				u_0,
				tau,
				TILE_SIZE*TILE_SIZE
			);
		} else {
			collision_step<<<grid, threads>>>(
				d_f,
				d_feq,
				d_rho,
				d_ux,
				d_uy,
				d_usqr,
				d_DR,
				d_WALL,
				d_FL,
				u_0,
				tau
			);
		}

		propagation_step<<<grid, threads>>>(
			d_f,
			d_fswap,
			nx,
			ny
		);

		// Swapping propagation buffers
		d_ftmp = d_fswap;
		d_fswap = d_f;
		d_f = d_ftmp;
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime,start,stop);
	}

	// Stop the timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(f, d_f, nx*ny*sizeof(directions_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(feq, d_feq, nx*ny*sizeof(directions_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(rho, d_rho, nx*ny*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(ux, d_ux, nx*ny*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, d_uy, nx*ny*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(usqr, d_usqr, nx*ny*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(DR, d_DR, nx*ny*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(WALL, d_WALL, nx*ny*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(FL, d_FL, nx*ny*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(mesh,d_mesh, nx*ny*sizeof(int),cudaMemcpyDeviceToHost);

	// print_matrix(usqr, nx*ny);
	std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;

	delete[] mesh;
	delete[] f;
	delete[] feq;
	delete[] rho;
	delete[] ux;
	delete[] uy;
	delete[] usqr;
	delete[] DR;
	delete[] WALL;
	delete[] FL;

	cudaFree(d_f);
	cudaFree(d_fswap);
	cudaFree(d_feq);
	cudaFree(d_rho);
	cudaFree(d_ux);
	cudaFree(d_uy);
	cudaFree(d_usqr);
	cudaFree(d_DR);
	cudaFree(d_WALL);
	cudaFree(d_FL);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	if (importfile) {
		free(matrix);
	}

	return 0;
}
