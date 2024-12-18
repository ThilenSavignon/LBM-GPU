#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream> // Pour debug ou affichage
#include "args.hxx" // Pour parser les arguments


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

template <typename T>
void printMatrix(T** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {         // Parcourt les lignes
		if (cols == 0) {
			std::cout << matrix[i];
			continue;
		}
        for (int j = 0; j < cols; ++j) {    // Parcourt les colonnes
            std::cout << matrix[i][j] << " "; // Affiche chaque élément
        }
        std::cout << std::endl;             // Saut de ligne après chaque ligne
    }
	if (cols == 0)
		std::cout << std::endl;                 // Saut de ligne supplémentaire à la fin
	std::cout << std::endl;
}

// fonction pour afficher toutes les données (rho, ux, uy, f, feq...) passées en paramètre
void printData(int nx, int ny, int iter, int Re, double rho_0, double u_0, double viscosity, double tau, int** mesh, double **f, double **feq, double **rho, double **ux, double **uy, double **usqr, bool **DR, bool **WL, bool **FL) {
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
	printMatrix(mesh, nx, ny);

	std::cout << "Affichage de  : f" << std::endl;
	printMatrix(f, nx*ny, 9);

	std::cout << "Affichage de  : feq" << std::endl;
	printMatrix(feq, nx*ny, 9);

	std::cout << "Affichage de  : rho" << std::endl;
	printMatrix(rho, nx*ny, 0);

	std::cout << "Affichage de  : ux" << std::endl;
	printMatrix(ux, nx*ny, 0);

	std::cout << "Affichage de  : uy" << std::endl;
	printMatrix(uy, nx*ny, 0);

	std::cout << "Affichage de  : usqr" << std::endl;
	printMatrix(usqr, nx*ny, 0);

	std::cout << "Affichage de  : DR" << std::endl;
	printMatrix(DR, nx, ny);

	std::cout << "Affichage de  : WALL" << std::endl;
	printMatrix(WL, nx, ny);

	std::cout << "Affichage de  : FL" << std::endl;
	printMatrix(FL, nx, ny);
}

__global__ void collision_step (
	double **f,
	double **feq,
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
		rho[INDEX] += f[INDEX][i];
	}

	// Macroscopic velocities
	ux[INDEX] = (DR[INDEX] ? u_0 : (f[INDEX][E] - f[INDEX][W] + f[INDEX][NE] + f[INDEX][SE] - f[INDEX][SW] - f[INDEX][NW]) / rho[INDEX]);
	uy[INDEX] = (DR[INDEX] ? 0 : (f[INDEX][N] - f[INDEX][S] + f[INDEX][NE] + f[INDEX][NW] - f[INDEX][SE] - f[INDEX][SW]) / rho[INDEX]);
	usqr[INDEX] = ux[INDEX] * ux[INDEX] + uy[INDEX] * uy[INDEX];

	feq[INDEX][C] = (4/9) * rho[INDEX] * (1. - 1.5 * usqr[INDEX]);
	feq[INDEX][E] = (1/9) * rho[INDEX] * (1 + 3 * ux[INDEX] + 4.5 * (ux[INDEX] * ux[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX][S] = (1/9) * rho[INDEX] * (1 - 3 * uy[INDEX] + 4.5 * (uy[INDEX] * uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX][W] = (1/9) * rho[INDEX] * (1 - 3 * ux[INDEX] + 4.5 * (ux[INDEX] * ux[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX][N] = (1/9) * rho[INDEX] * (1 + 3 * uy[INDEX] + 4.5 * (uy[INDEX] * uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX][NE] = (1/36) * rho[INDEX] * (1 + 3 * (ux[INDEX] + uy[INDEX]) + 4.5 * (ux[INDEX] + uy[INDEX]) * (ux[INDEX] + uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX][NE] = (1/36) * rho[INDEX] * (1 + 3 * (ux[INDEX] - uy[INDEX]) + 4.5 * (ux[INDEX] - uy[INDEX]) * (ux[INDEX] - uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX][NE] = (1/36) * rho[INDEX] * (1 + 3 * (-ux[INDEX] - uy[INDEX]) + 4.5 * (-ux[INDEX] - uy[INDEX]) * (-ux[INDEX] - uy[INDEX]) - 1.5 * usqr[INDEX]);
	feq[INDEX][NE] = (1/36) * rho[INDEX] * (1 + 3 * (-ux[INDEX] + uy[INDEX]) + 4.5 * (-ux[INDEX] + uy[INDEX]) * (-ux[INDEX] + uy[INDEX]) - 1.5 * usqr[INDEX]);

	if(WL[INDEX]) {
		f[INDEX][E] = f[INDEX][W];
		f[INDEX][S] = f[INDEX][N];
		f[INDEX][W] = f[INDEX][E];
		f[INDEX][N] = f[INDEX][S];
		f[INDEX][NE] = f[INDEX][SW];
		f[INDEX][SE] = f[INDEX][NW];
		f[INDEX][SW] = f[INDEX][NE];
		f[INDEX][NW] = f[INDEX][SE];
	} else if (DR[INDEX]) {
		f[INDEX] = feq[INDEX];
	} else {
		for (int i=0; i<9; i++) {
			f[INDEX][i] = f[INDEX][i] * (1. - 1. / tau) + feq[INDEX][i] / tau;
		}
	}
}

__global__ void propagation_step(double **f_src, double **f_dst, int nx, int ny) {
	if(INDEX_X < nx-1)	f_dst[INDEX_FROM(INDEX_X + 1, INDEX_Y)][E] = f_src[INDEX][E];
	if(INDEX_X > 0)		f_dst[INDEX_FROM(INDEX_X - 1, INDEX_Y)][W] = f_src[INDEX][W];
	if(INDEX_Y < ny-1)	f_dst[INDEX_FROM(INDEX_X, INDEX_Y + 1)][S] = f_src[INDEX][S];
	if(INDEX_Y > 0)		f_dst[INDEX_FROM(INDEX_X, INDEX_Y - 1)][N] = f_src[INDEX][N];
	if(INDEX_X < nx-1 && INDEX_Y < ny-1)	f_dst[INDEX_FROM(INDEX_X + 1, INDEX_Y + 1)][SE] = f_src[INDEX][SE];
	if(INDEX_X < nx-1 && INDEX_Y > 0)		f_dst[INDEX_FROM(INDEX_X + 1, INDEX_Y - 1)][NE] = f_src[INDEX][NE];
	if(INDEX_X > 0 && INDEX_Y < ny-1)		f_dst[INDEX_FROM(INDEX_X - 1, INDEX_Y + 1)][SW] = f_src[INDEX][SW];
	if(INDEX_X > 0 && INDEX_Y > 0)			f_dst[INDEX_FROM(INDEX_X - 1, INDEX_Y - 1)][NW] = f_src[INDEX][NW];

}

int main (int argc, char** argv){

	// Define parser
	args::ArgumentParser parser("main", "This is a main program for the LBM simulation");

	// Set parser value
	args::Flag print(parser, "print", "print the matrix at the end", {"p"});
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
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
    int nx, ny, iter, Re;
    if (width && height) { //initialisation de la taille de la grille
		nx = args::get(width);
		ny = args::get(height);
	} else {
		nx = 32;
		ny = 32;
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
    int** mesh; // 0 = fluid, 1 = wall, 2 = driving fluid
	mesh = new int*[nx];
	for(int i = 0; i<nx; i++){
		mesh[i] = new int[ny];
	}
    for(int i = 0; i<nx; i++){
        for(int j = 0; j<ny; j++){
            if(i == 0)
                mesh[i][j]=2; // premiere ligne est un driving fluid
            else if (j == 0 || j == ny-1 || i == nx-1)
                mesh[i][j]=1; // les extremes sont des murs
            else
                mesh[i][j] = 0; // le reste est vide
        }
    }
    
    double **f, **feq, **rho, **ux, **uy, **usqr;
	f = new double*[nx*ny]; // distribution function
	feq = new double*[nx*ny]; // equilibrium distribution function
	rho = new double*[nx*ny]; // macroscopic density
	ux = new double*[nx*ny]; // macroscopic velocity in direction x
	uy = new double*[nx*ny]; // macroscopic velocity in direction y
	usqr = new double*[nx*ny]; // helper variable
	for(int i = 0; i<nx*ny; i++){
		f[i] = new double[9];
		feq[i] = new double[9];
	}
    for(int i = 0; i<nx*ny; i++){
        rho[i]=0; 
        ux[i]=0; 
        uy[i]=0; 
        usqr[i]=0;
        for (int j = 0; j<9; j++){
            f[i][j] = 0.0; // distribution function values for each cell
            feq[i][j] = 0.0; // equilibrium distribution function value
        }
    }

	bool **DR, **WALL, **FL;
	DR = new bool*[nx]; // driving fluid
	WALL = new bool*[nx]; // wall
	FL = new bool*[nx]; // fluid
	for(int i = 0; i<nx; i++){
		DR[i] = new bool[ny];
		WALL[i] = new bool[ny];
		FL[i] = new bool[ny];
	}

	for(int i = 0; i<nx; i++){
        for(int j = 0; j<ny; j++){
			if(mesh[i][j]==0){
				
				FL[i][j] = true;
				WALL[i][j] = false; // Store the flattened index
				DR[i][j] = false; // Store the flattened index
				
			}else if(mesh[i][j]==1){
				FL[i][j] = false;
				WALL[i][j] = true; // Store the flattened index
				DR[i][j] = false; // Store the flattened index
			}else if (mesh[i][j]==2){
				FL[i][j] = false;
				WALL[i][j] = false; // Store the flattened index
				DR[i][j] = true; // Store the flattened index
			}
		}
	}

	if (print) {
		printData(nx, ny, iter, Re, rho_0, u_0, viscosity, tau, mesh, f, feq, rho, ux, uy, usqr, DR, WALL, FL);
	}

	// free memory
	for(int i = 0; i<nx; i++){
		delete[] mesh[i];
		delete[] DR[i];
		delete[] WALL[i];
		delete[] FL[i];
	}

	for(int i = 0; i<nx*ny; i++){
		delete[] f[i];
		delete[] feq[i];
	}

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
