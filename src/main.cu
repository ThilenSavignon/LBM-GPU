#include <iostream>


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

template <typename T>
void printMatrix(T** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {         // Parcourt les lignes
        for (int j = 0; j < cols; ++j) {    // Parcourt les colonnes
            std::cout << matrix[i][j] << " "; // Affiche chaque élément
        }
        std::cout << std::endl;             // Saut de ligne après chaque ligne
    }
}

int main (int argc, char** argv){

    // initialisation des parametres de la simulation
    int nx, ny, iter, Re;
    nx = 32;
    ny = 32;
    iter = 3000;
	Re=1000;

    // initialisation des variables
    double  rho_0, u_0, viscosity, tau;
    rho_0 = 1;
    u_0 = 0.1;
    tau = (6*viscosity+1)/2;

    // initialisation de la grille de la simulation
    int** mesh;
    for(int i = 0; i<nx; i++){
        for(int j = 0; j<ny; j++){
            if(i == 0)
                mesh[i][j]=2; // premiere ligne est un driving fluid
            else if (j == 0 || j == ny-1 || i == nx-1)
                mesh[i][j]=1; // les extremes sont des murs
            else
                mesh[i][j] = 0; // le reste est vide
            // std::cout << mesh[i][j];
        }
        // std::cout << std::endl;
    }
    std::cout << "Affichage de  : mesh" << std::endl;
    printMatrix(mesh, nx, ny);
    
    double **f, **feq, **rho, **ux, **uy, **usqr;
    for(int i = 0; i<nx*ny; i++){
        rho[i]=0; // macroscopic density
        ux[i]=0; // macroscopic velocity in direction x
        uy[i]=0; // macroscopic velocity in direction y
        usqr[i]=0; // helper variable
        for (int j = 0; j<9; j++){
            f[i][j] = 0.0; // distribution function values for each cell
            feq[i][j] = 0.0; // equilibrium distribution function value
        }
    }
}

__global__ void find_values(int *FL,int *WALL,int *DR, int **mesh, int nx, int ny, int *counter){
	int x = INDEX / ny;
	int y = INDEX % nx;
	if(mesh[x][y]==0){
		int pos = atomicAdd(counter, 1); // Atomic increment to get unique position
		FL[pos] = x * ny + y; // Store the flattened index
	}else if(mesh[x][y]==1){
		int pos = atomicAdd(counter, 1); // Atomic increment to get unique position
		WALL[pos] = x * ny + y; // Store the flattened index
	}else if (mesh[x][y]==2){
		int pos = atomicAdd(counter, 1); // Atomic increment to get unique position
		DR[pos] = x * ny + y; // Store the flattened index
	}
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