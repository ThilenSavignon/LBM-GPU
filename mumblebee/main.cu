#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream> // Pour debug ou affichage
#include "args.hxx" // Pour parser les arguments
#include <assert.h> // Pour les assertions


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

#define INDEX (gridDim.x * blockDim.x * (blockDim.y * blockIdx.y + threadIdx.y) + blockIdx.x * blockDim.x + threadIdx.x)
#define INDEX_X (blockIdx.x * blockDim.x + threadIdx.x)
#define INDEX_Y	(blockDim.y * blockIdx.y + threadIdx.y)
#define OFFSET_Y (gridDim.x * blockDim.x)
#define INDEX_FROM(x, y) ((OFFSET_Y) * (y) + (x))

int nx = 32;
int ny = 32;

typedef union Directions {
	double direction[9];
	struct {
		double c;
		double e;
		double s;
		double w;
		double n;
		double ne;
		double se;
		double sw;
		double nw;
	};
} directions_t;

template <typename T>
void print(T* table, int size) {
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
void printData(int nx, int ny, int iter, int Re, double rho_0, double u_0, double viscosity, double tau, int* mesh, directions_t *f, directions_t *feq, double *rho, double *ux, double *uy, double *usqr, bool *DR, bool *WL, bool *FL) {
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
	print(mesh, nx*ny);

	std::cout << "Affichage de  : f" << std::endl;
	printdirection(f, nx, ny);

	std::cout << "Affichage de  : feq" << std::endl;
	printdirection(feq, nx, ny);

	std::cout << "Affichage de  : rho" << std::endl;
	print(rho, nx*ny);

	std::cout << "Affichage de  : ux" << std::endl;
	print(ux, nx*ny);

	std::cout << "Affichage de  : uy" << std::endl;
	print(uy, nx*ny);

	std::cout << "Affichage de  : usqr" << std::endl;
	print(usqr, nx*ny);

	std::cout << "Affichage de  : DR" << std::endl;
	print(DR, nx*ny);

	std::cout << "Affichage de  : WALL" << std::endl;
	print(WL, nx*ny);

	std::cout << "Affichage de  : FL" << std::endl;
	print(FL, nx*ny);
}

__global__ void collision_step (
	directions_t *f,
	directions_t *feq,
	double *rho,
	double *ux,
	double *uy,
	double *usqr,
	bool *DR,
	bool *WL,
	bool *FL,
	double u_0,
	double tau) {
	
	// Macroscopic density
	rho[INDEX] = 0;
	for (int i=0; i<9; i++) {
		rho[INDEX] += f[INDEX].direction[i];
	}

	// Macroscopic velocities
	ux[INDEX] = (DR[INDEX] ? u_0 : (f[INDEX].e - f[INDEX].w + f[INDEX].ne + f[INDEX].se - f[INDEX].sw - f[INDEX].nw) / rho[INDEX]);
	uy[INDEX] = (DR[INDEX] ? 0 : (f[INDEX].n - f[INDEX].s + f[INDEX].ne + f[INDEX].nw - f[INDEX].se - f[INDEX].sw) / rho[INDEX]);
	usqr[INDEX] = ux[INDEX] * ux[INDEX] + uy[INDEX] * uy[INDEX];

	feq[INDEX].c = (4/9) * rho[INDEX] * (1. - 1.5 * usqr[INDEX]);
	feq[INDEX].e = (1/9) * rho[INDEX] * (1 + 3 * ux[INDEX] + 4.5 * (ux[INDEX] * ux[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].s = (1/9) * rho[INDEX] * (1 - 3 * uy[INDEX] + 4.5 * (uy[INDEX] * uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].w = (1/9) * rho[INDEX] * (1 - 3 * ux[INDEX] + 4.5 * (ux[INDEX] * ux[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].n = (1/9) * rho[INDEX] * (1 + 3 * uy[INDEX] + 4.5 * (uy[INDEX] * uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].ne = (1/36) * rho[INDEX] * (1 + 3 * (ux[INDEX] + uy[INDEX]) + 4.5 * (ux[INDEX] + uy[INDEX]) * (ux[INDEX] + uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].se = (1/36) * rho[INDEX] * (1 + 3 * (ux[INDEX] - uy[INDEX]) + 4.5 * (ux[INDEX] - uy[INDEX]) * (ux[INDEX] - uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].sw = (1/36) * rho[INDEX] * (1 + 3 * (-ux[INDEX] - uy[INDEX]) + 4.5 * (-ux[INDEX] - uy[INDEX]) * (-ux[INDEX] - uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX].nw = (1/36) * rho[INDEX] * (1 + 3 * (-ux[INDEX] + uy[INDEX]) + 4.5 * (-ux[INDEX] + uy[INDEX]) * (-ux[INDEX] + uy[INDEX]) - 1.5 * usqr[INDEX]);

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
		f[INDEX] = feq[INDEX];
	} else {
		for (int i=0; i<9; i++) {
			f[INDEX].direction[i] = f[INDEX].direction[i] * (1. - 1. / tau) + feq[INDEX].direction[i] / tau;
		}
	}
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

int main (int argc, char** argv){

	// Define parser
	args::ArgumentParser parser("main", "This is a main program for the LBM simulation");

	// Set parser value
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::Flag print(parser, "print", "print the matrix at the end", {"p"});
	args::ValueFlag<int> width(parser, "width", "Width of matrix ", {"w"}, 32);
	args::ValueFlag<int> height(parser, "height", "Height of matrix ", {"h"},32);
	args::ValueFlag<int> iterations(parser, "iteration", "Number of iterations", {"i"}, 3000);
	// args::Flag is_shared(parser, "shared", "Use shared memory", {"s"});

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
		iter = 3000;
	Re=1000; // nombre de Reynolds

    // initialisation des variables
    double  rho_0, u_0, viscosity, tau;
    rho_0 = 1; // densite initiale
    u_0 = 0.1; // vitesse initiale
	viscosity = (ny-1)*u_0/Re; // viscosite
    tau = (6*viscosity+1)/2; // relaxation time
	

    // initialisation de la grille de la simulation
    int* mesh = new int[nx*ny]; // 0 = fluid, 1 = wall, 2 = driving fluid
    for(long i = 0; i<nx*ny; i++){
		if(i < ny)
			mesh[i]=2; // le fluide est injecté à gauche
		else if (i % ny == 0 || i % ny == ny-1 || i > nx*ny-ny)
			mesh[i] = 1; // les bords sont des murs
		else
			mesh[i] = 0; // le reste est du fluide
    }
    
    directions_t *f, *feq;
	double *rho, *ux, *uy, *usqr;
	f = new directions_t[nx*ny]; // distribution function
	feq = new directions_t[nx*ny]; // equilibrium distribution function
	rho = new double[nx*ny]; // macroscopic density
	ux = new double[nx*ny]; // macroscopic velocity in direction x
	uy = new double[nx*ny]; // macroscopic velocity in direction y
	usqr = new double[nx*ny]; // helper variable
	for(long i = 0; i<nx*ny; i++){
		for(int j = 0; j<9; j++) {
			feq[i].direction[j] = 0;
		}

		f[i].c = rho_0 * 4 / 9;
		f[i].e = rho_0 / 9;
		f[i].s = rho_0 / 9;
		f[i].w = rho_0 / 9;
		f[i].n = rho_0 / 9;
		f[i].ne = rho_0 / 36;
		f[i].se = rho_0 / 36;
		f[i].sw = rho_0 / 36;
		f[i].nw = rho_0 / 36;

        rho[i]=0; 
        ux[i]=0; 
        uy[i]=0; 
        usqr[i]=0;
    }

	bool *DR, *WALL, *FL;
	DR = new bool[nx*ny]; // driving fluid
	WALL = new bool[nx*ny]; // wall
	FL = new bool[nx*ny]; // fluid

	for(long i = 0; i<nx*ny; i++){
		if(mesh[i]==0){
			FL[i] = true;
			WALL[i] = false; // Store the flattened index
			DR[i] = false; // Store the flattened index
			
		}else if(mesh[i]==1){
			FL[i] = false;
			WALL[i] = true; // Store the flattened index
			DR[i] = false; // Store the flattened index
		}else if (mesh[i]==2){
			FL[i] = false;
			WALL[i] = false; // Store the flattened index
			DR[i] = true; // Store the flattened index
		}
	}

	if (print) {
		printData(nx, ny, iter, Re, rho_0, u_0, viscosity, tau, mesh, f, feq, rho, ux, uy, usqr, DR, WALL, FL);
	}

	//================= CUDA =================
	directions_t *d_f, *d_fswap, *d_ftmp, *d_feq;
	double *d_rho, *d_ux, *d_uy, *d_usqr;
	bool *d_DR, *d_WALL, *d_FL;

	cudaMalloc(&d_f, nx*ny*sizeof(directions_t));
	cudaMalloc(&d_fswap, nx*ny*sizeof(directions_t));
	cudaMalloc(&d_feq, nx*ny*sizeof(directions_t));
	cudaMalloc(&d_rho, nx*ny*sizeof(double));
	cudaMalloc(&d_ux, nx*ny*sizeof(double));
	cudaMalloc(&d_uy, nx*ny*sizeof(double));
	cudaMalloc(&d_usqr, nx*ny*sizeof(double));
	cudaMalloc(&d_DR, nx*sizeof(bool));
	cudaMalloc(&d_WALL, nx*sizeof(bool));
	cudaMalloc(&d_FL, nx*sizeof(bool));

	cudaMemcpy(d_f, f, nx*ny*sizeof(directions_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fswap, f, nx*ny*sizeof(directions_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_feq, feq, nx*ny*sizeof(directions_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, rho, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ux, ux, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_uy, uy, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_usqr, usqr, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_DR, DR, nx*ny*sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_WALL, WALL, nx*ny*sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FL, FL, nx*ny*sizeof(bool), cudaMemcpyHostToDevice);

	//============ MAIN LOOP =============

	dim3 dimBlock(32, 32);
	dim3 dimGrid(1,1);

	for(int i=0; i<iter; i++) {
		collision_step<<<dimGrid, dimBlock>>>(
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

		propagation_step<<<dimGrid, dimBlock>>>(
			d_f,
			d_fswap,
			nx,
			ny
		);

		// Swapping propagation buffers
		d_ftmp = d_fswap;
		d_fswap = d_f;
		d_f = d_ftmp;
	}

	cudaMemcpy(f, d_f, nx*ny*sizeof(directions_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(feq, d_feq, nx*ny*sizeof(directions_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(rho, d_rho, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ux, d_ux, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, d_uy, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(usqr, d_usqr, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(DR, d_DR, nx*ny*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(WALL, d_WALL, nx*ny*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(FL, d_FL, nx*ny*sizeof(bool), cudaMemcpyDeviceToHost);
	
	printdirection(f, nx, ny);
	//=============== END ================

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

	return 0;
}
