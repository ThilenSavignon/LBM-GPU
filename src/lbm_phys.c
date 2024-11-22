/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#include <assert.h>
#include <stdlib.h>
#include "lbm_config.h"
#include "lbm_struct.h"
#include "lbm_phys.h"
#include "lbm_comm.h"

/****************************************************/
/**
 * Definitions des 9 vecteurs de base utilisé pour discrétiser les directions sur chaque mailles.
**/
#if DIRECTIONS == 9 && DIMENSIONS == 2
const Vector direction_matrix[DIRECTIONS] = {
	{+0.0,+0.0},
	{+1.0,+0.0}, {+0.0,+1.0}, {-1.0,+0.0}, {+0.0,-1.0},
	{+1.0,+1.0}, {-1.0,+1.0}, {-1.0,-1.0}, {+1.0,-1.0}
};
#else
#error Need to defined adapted direction matrix.
#endif

/****************************************************/
/**
 * Poids utilisé pour compenser les différentes de longueur des 9 vecteurs directions.
**/
#if DIRECTIONS == 9
const double equil_weight[DIRECTIONS] = {
	4.0/9.0 ,
	1.0/9.0 , 1.0/9.0 , 1.0/9.0 , 1.0/9.0,
	1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
//opposite directions, for bounce back implementation
const int opposite_of[DIRECTIONS] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
#else
#error Need to defined adapted equibirium distribution function
#endif

/****************************************************/
/**
 * Renvoie le résultat du produit des deux vecteurs passés en paramêtre.
**/
double lbm_phys_vect_norme_2(const Vector vect1,const Vector vect2)
{
	//vars
	int k;
	double res = 0.0;

	//loop on dimensions
	for ( k = 0 ; k < DIMENSIONS ; k++)
		res += vect1[k] * vect2[k];

	return res;
}

/****************************************************/
/**
 * Calcule la densité macroscopiques de la cellule en sommant ses DIRECTIONS
 * densités microscopiques.
**/
double lbm_phys_cell_density(const lbm_mesh_cell_t cell)
{
	//vars
	int k;
	double res = 0.0;

	//errors
	assert( cell != NULL );

	//loop on directions
	for( k = 0 ; k < DIRECTIONS ; k++)
		res += cell[k];

	//return res
	return res;
}

/****************************************************/
/**
 * Calcule la vitesse macroscopiques de la cellule en sommant ses DIRECTIONS
 * densités microscopiques.
 * @param cell_density Densité macroscopique de la cellules.
**/
void lbm_phys_cell_velocity(Vector v,const lbm_mesh_cell_t cell,double cell_density)
{
	//vars
	int k,d;

	//errors
	assert(v != NULL);
	assert(cell != NULL);

	//loop on all dimensions
	for ( d = 0 ; d < DIMENSIONS ; d++)
	{
		//reset value
		v[d] = 0.0;

		//sum all directions
		for ( k = 0 ; k < DIRECTIONS ; k++)
			v[d] += cell[k] * direction_matrix[k][d];

		//normalize
		v[d] = v[d] / cell_density;
	}
}

/****************************************************/
/**
 * Calcule le profile de densité microscopique à l'équilibre pour une direction donnée.
 * C'est ici qu'intervient un dérivé de navier-stokes.
 * @param velocity Vitesse macroscopique du fluide sur la maille.
 * @param density Densité macroscopique du fluide sur la maille.
 * @param direction Direction pour laquelle calculer l'état d'équilibre.
**/
double lbm_phys_equilibrium_profile(Vector velocity,double density,int direction)
{
	//vars
	double v2;
	double p;
	double p2;
	double feq;

	//velocity norme 2 (v * v)
	v2 = lbm_phys_vect_norme_2(velocity,velocity);

	//calc e_i * v_i / c
	p = lbm_phys_vect_norme_2(direction_matrix[direction],velocity);
	p2 = p * p;

	//terms without density and direction weight
	feq = 1.0
		+ (3.0 * p)
		+ ((9.0 / 2.0) * p2)
		- ((3.0 / 2.0) * v2);

	//mult all by density and direction weight
	feq *= equil_weight[direction] * density;

	return feq;
}

/****************************************************/
/**
 * Calcule le vecteur de lbm_phys_collision entre les fluides de chacune des directions.
**/
void lbm_phys_cell_collision(lbm_mesh_cell_t cell_out,const lbm_mesh_cell_t cell_in)
{
	//vars
	int k;
	double density;
	Vector v;
	double feq;

	//compute macroscopic values
	density = lbm_phys_cell_density(cell_in);
	lbm_phys_cell_velocity(v,cell_in,density);

	//loop on microscopic directions
	for( k = 0 ; k < DIRECTIONS ; k++)
	{
		//compute f at equilibr.
		feq = lbm_phys_equilibrium_profile(v,density,k);
		//compute fout
		cell_out[k] = cell_in[k] - RELAX_PARAMETER * (cell_in[k] - feq);
	}
}

/****************************************************/
/**
 * Applique une reflexion sur les différentes directions pour simuler la présence d'un solide.
**/
void lbm_phys_bounce_back(lbm_mesh_cell_t cell)
{
	//vars
	int k;
	double tmp[DIRECTIONS];

	//compute bounce back
	for ( k = 0 ; k < DIRECTIONS ; k++)
		tmp[k] = cell[opposite_of[k]];

	//compute bounce back
	for ( k = 0 ; k < DIRECTIONS ; k++)
		cell[k] = tmp[k];
}

/****************************************************/
/**
 * Fournit la vitesse de poiseuille pour une position donnée en considérant un tube de taille donnée.
 * @param i Position pour laquelle on cherche la vitesse.
 * @param size diamètre du tube.
**/
double lbm_phys_poiseuille(int i,int size)
{
	double y = (double)(i - 1);
	double L = (double)(size - 1);
	return 4.0 * INFLOW_MAX_VELOCITY / ( L * L ) * ( L * y - y * y );
}

/****************************************************/
/**
 * Applique la méthode de Zou/He pour simler un fluidre entrant dans le domain de gauche vers la droite sur une
 * interface verticale. Le profile de vitesse du fluide entrant suit une distribution de poiseuille.
 * @param mesh Maillage considéré (surtout pour avoir la hauteur.)
 * @param cell Maille à mettre à jour.
 * @param id_y Position en y de la cellule pour savoir comment calculer la vitesse de poiseuille.
**/
void lbm_phys_inflow_zou_he_poiseuille_distr( const lbm_mesh_t * mesh, lbm_mesh_cell_t cell,int id_y)
{
	//vars
	double v;
	double density;

	//errors
	#if DIRECTIONS != 9
	#error Implemented only for 9 directions
	#endif

	//set macroscopic fluide info
	//poiseuille distr on X and null on Y
	//we just want the norm, so v = v_x
	v = lbm_phys_poiseuille(id_y,mesh->height);

	//compute rho from u and inner flow on surface
	density = (cell[0] + cell[2] + cell[4] + 2 * ( cell[3] + cell[6] + cell[7] )) / (1.0 - v) ;

	//now compute unknown microscopic values
	cell[1] = cell[3];// + (2.0/3.0) * density * v_y <--- no velocity on Y so v_y = 0
	cell[5] = cell[7] - (1.0/2.0) * (cell[2] - cell[4])
	                         + (1.0/6.0) * (density * v);
	                       //+ (1.0/2.0) * density * v_y    <--- no velocity on Y so v_y = 0
	cell[8] = cell[6] + (1.0/2.0) * (cell[2] - cell[4])
	                         + (1.0/6.0) * (density * v);
	                       //- (1.0/2.0) * density * v_y    <--- no velocity on Y so v_y = 0

	//no need to copy already known one as the value will be "loss" in the wall at propagatation time
}

/****************************************************/
/**
 * Applique la méthode de Zou/He pour simler un fluidre sortant du domain de gauche vers la droite sur une
 * interface verticale. La condition appliquer pour construire l'équation est le maintient d'un gradiant de densité
 * nulle à l'interface.
 * @param mesh Maillage considéré (surtout pour avoir la hauteur.)
 * @param cell Maille à mettre à jour
 * @param id_y Position en y de la cellule pour savoir comment calculer la vitesse de poiseuille.
**/
void lbm_phys_outflow_zou_he_const_density(lbm_mesh_cell_t cell)
{
	//vars
	const double density = 1.0;
	double v;

	//errors
	#if DIRECTIONS != 9
	#error Implemented only for 9 directions
	#endif

	//compute macroscopic v depeding on inner flow going onto the wall
	v = -1.0 + (1.0 / density) * (cell[0] + cell[2] + cell[4] + 2 * (cell[1] + cell[5] + cell[8]));

	//now can compute unknown microscopic values
	cell[3] = cell[1] - (2.0/3.0) * density * v;
	cell[7] = cell[5] + (1.0/2.0) * (cell[2] - cell[4])
	                       //- (1.0/2.0) * (density * v_y)    <--- no velocity on Y so v_y = 0
	                         - (1.0/6.0) * (density * v);
	cell[6] = cell[8] + (1.0/2.0) * (cell[4] - cell[2])
	                       //+ (1.0/2.0) * (density * v_y)    <--- no velocity on Y so v_y = 0
	                         - (1.0/6.0) * (density * v);
}

/****************************************************/
void lbm_phys_special_cells_one_cell(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm,int i,int j)
{
	switch (*( lbm_cell_type_t_get_cell( mesh_type , i, j) ))
	{
		case CELL_FUILD:
			break;
		case CELL_BOUNCE_BACK:
			lbm_phys_bounce_back(lbm_mesh_get_cell(mesh, i, j));
			break;
		case CELL_LEFT_IN:
			lbm_phys_inflow_zou_he_poiseuille_distr(mesh, lbm_mesh_get_cell(mesh, i, j) ,j + comm->y);
			break;
		case CELL_RIGHT_OUT:
			lbm_phys_outflow_zou_he_const_density(lbm_mesh_get_cell(mesh, i, j));
			break;
	}
}

/****************************************************/
/**
 * Applique les actions spéciale liée aux conditions de bords ou au réflexions sur l'obstacle.
**/
void lbm_phys_special_cells(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm)
{
	//vars
	int i,j;

	//loop on all inner cells
	for( i = 0 ; i < mesh->width ; i++ )
		for( j = 0 ; j < mesh->height  ; j++)
			lbm_phys_special_cells_one_cell(mesh,mesh_type,comm,i,j);
}

/****************************************************/
/**
 * Applique les actions spéciale liée aux conditions de bords ou au réflexions sur l'obstacle.
**/
void lbm_phys_special_cells_inner(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm)
{
	//vars
	int i,j;

	//loop on all inner cells
	for( i = 1 ; i < mesh->width - 1 ; i++ )
		for( j = 1 ; j < mesh->height - 1 ; j++)
			lbm_phys_special_cells_one_cell(mesh,mesh_type,comm,i,j);
}

/****************************************************/
/**
 * Applique les actions spéciale liée aux conditions de bords ou au réflexions sur l'obstacle.
**/
void lbm_phys_special_cells_border(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm)
{
	//vars
	int i,j;

	//top
	for ( i = 0 ; i < mesh->width; i++)
		lbm_phys_special_cells_one_cell(mesh,mesh_type,comm,i,0);

	//bottom
	for ( i = 0 ; i < mesh->width; i++)
		lbm_phys_special_cells_one_cell(mesh,mesh_type,comm,i,mesh->height - 1);

	//left
	for ( j = 0 ; j < mesh->height ; j++)
		lbm_phys_special_cells_one_cell(mesh,mesh_type,comm,0,j);

	//right
	for ( j = 0 ; j < mesh->height ; j++)
		lbm_phys_special_cells_one_cell(mesh,mesh_type,comm,mesh->width - 1,j);
}

/****************************************************/
/**
 * Calcule les collision sur chacune des cellules.
 * @param mesh Maillage sur lequel appliquer le calcule.
**/
void lbm_phys_collision(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in)
{
	//vars
	int i,j;

	//errors
	assert(mesh_in->width == mesh_out->width);
	assert(mesh_in->height == mesh_out->height);

	//loop on all inner cells
	//to avoid reflexion of first shock wave : i = 1
	for( i = 0 ; i < mesh_in->width ; i++ )
		for( j = 0 ; j < mesh_in->height ; j++)
			lbm_phys_cell_collision(lbm_mesh_get_cell(mesh_out, i, j),lbm_mesh_get_cell(mesh_in, i, j));
}

/****************************************************/
/**
 * Calcule les collision sur chacune des cellules.
 * @param mesh Maillage sur lequel appliquer le calcule.
**/
void lbm_phys_collision_inner(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in)
{
	//vars
	int i,j;

	//errors
	assert(mesh_in->width == mesh_out->width);
	assert(mesh_in->height == mesh_out->height);

	//loop on all inner cells
	//to avoid reflexion of first shock wave : i = 1
	for( i = 1 ; i < mesh_in->width - 1 ; i++ )
		for( j = 1 ; j < mesh_in->height - 1 ; j++)
			lbm_phys_cell_collision(lbm_mesh_get_cell(mesh_out, i, j),lbm_mesh_get_cell(mesh_in, i, j));
}

/****************************************************/
/**
 * Calcule les collision sur chacune des cellules.
 * @param mesh Maillage sur lequel appliquer le calcule.
**/
void lbm_phys_collision_border(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in)
{
	//vars
	int i,j;

	//errors
	assert(mesh_in->width == mesh_out->width);
	assert(mesh_in->height == mesh_out->height);

	//top
	for ( i = 0 ; i < mesh_out->width; i++)
		lbm_phys_cell_collision(lbm_mesh_get_cell(mesh_out, i, 0),lbm_mesh_get_cell(mesh_in, i, 0));

	//bottom
	for ( i = 0 ; i < mesh_out->width; i++)
		lbm_phys_cell_collision(lbm_mesh_get_cell(mesh_out, i, mesh_out->height - 1),lbm_mesh_get_cell(mesh_in, i, mesh_out->height - 1));

	//left
	for ( j = 0 ; j < mesh_out->height ; j++)
		lbm_phys_cell_collision(lbm_mesh_get_cell(mesh_out, 0, j),lbm_mesh_get_cell(mesh_in, 0, j));

	//right
	for ( j = 0 ; j < mesh_out->height ; j++)
		lbm_phys_cell_collision(lbm_mesh_get_cell(mesh_out, mesh_out->width - 1, j),lbm_mesh_get_cell(mesh_in, mesh_out->width - 1, j));
}

/****************************************************/
void lbm_phys_propagation_one_cell(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in,int i, int j)
{
	int k;
	int ii,jj;

	//for all direction
	for ( k  = 0 ; k < DIRECTIONS ; k++)
	{
		//compute destination point
		ii = (i + direction_matrix[k][0]);
		jj = (j + direction_matrix[k][1]);
		//propagate to neighboor nodes
		if ((ii >= 0 && ii < mesh_out->width) && (jj >= 0 && jj < mesh_out->height))
			lbm_mesh_get_cell(mesh_out, ii, jj)[k] = lbm_mesh_get_cell(mesh_in, i, j)[k];
	}
}

/****************************************************/
/**
 * lbm_phys_propagation des densité vers les maillse voisines.
 * @param mesh_out Maillage de sortie.
 * @param mesh_in Maillage d'entrée (ne doivent pas être les mêmes).
**/
void lbm_phys_propagation(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in)
{
	//vars
	int i,j;

	//loop on all cells
	for ( i = 0 ; i < mesh_out->width; i++)
		for ( j = 0 ; j < mesh_out->height ; j++)
			lbm_phys_propagation_one_cell(mesh_out,mesh_in,i,j);
}

/****************************************************/
/**
 * lbm_phys_propagation des densité vers les maillse voisines.
 * @param mesh_out Maillage de sortie.
 * @param mesh_in Maillage d'entrée (ne doivent pas être les mêmes).
**/
void lbm_phys_propagation_inner(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in)
{
	//vars
	int i,j;

	//loop on all cells
	for ( i = 1 ; i < mesh_out->width - 1; i++)
		for ( j = 1 ; j < mesh_out->height - 1; j++)
			lbm_phys_propagation_one_cell(mesh_out,mesh_in,i,j);
}

/****************************************************/
/**
 * lbm_phys_propagation des densité vers les maillse voisines.
 * @param mesh_out Maillage de sortie.
 * @param mesh_in Maillage d'entrée (ne doivent pas être les mêmes).
**/
void lbm_phys_propagation_border(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in)
{
	//vars
	int i,j;

	//loop on all cells
	for ( i = 0 ; i < mesh_out->width; i++)
		lbm_phys_propagation_one_cell(mesh_out,mesh_in,i,0);

	for ( i = 0 ; i < mesh_out->width; i++)
		lbm_phys_propagation_one_cell(mesh_out,mesh_in,i,mesh_out->height - 1);

	for ( j = 0 ; j < mesh_out->height ; j++)
		lbm_phys_propagation_one_cell(mesh_out,mesh_in,0,j);

	for ( j = 0 ; j < mesh_out->height ; j++)
		lbm_phys_propagation_one_cell(mesh_out,mesh_in,mesh_out->width - 1,j);
}
