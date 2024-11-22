/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
//std
#include <assert.h>
//mpi
#include <mpi.h>
//internal
#include "lbm_phys.h"
#include "lbm_init.h"
//magick want for image processing
#ifdef HAVE_MAGICK_WAND
	#include <wand/MagickWand.h>
#endif

/****************************************************/
/**
 * Applique les conditions initiale avec un fluide au repos. Utiliser uniquement
 * pour des test lors de l'implémentation.
**/
//Apply initial conditions :
// - v = 0
// - rho = 1
//Equation is : f_equilibre(x,t) = w_i * rho + rho * s_i(v(x,t))
//Here v = 0 so s_i(*) = 0, rho = 1, so keep only w_i
//w_i is the direction weight.
void lbm_init_velocity_0_density_1(lbm_mesh_t * mesh)
{
	//vars
	int i,j,k;

	//errors
	assert(mesh != NULL);

	//loop on all cells
	for ( i = 0 ; i <  mesh->width ; i++)
		for ( j = 0 ; j <  mesh->height ; j++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
				lbm_mesh_get_cell(mesh, i, j)[k] = equil_weight[k];
}

/****************************************************/
/**
 * Initialisation de l'obstacle, on bascule les types des mailles associé à CELL_BOUNCE_BACK.
 * Ici l'obstacle est un cercle de centre (OBSTACLE_X,OBSTACLE_Y) et de rayon OBSTACLE_R.
**/
void lbm_init_circle_obstacle(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm)
{
	//vars
	int i,j;

	//loop on nodes
	for ( i =  comm->x; i < mesh->width + comm->x ; i++)
	{
		for ( j =  comm->y ; j <  mesh->height + comm->y ; j++)
		{
			if ( ( (i-OBSTACLE_X) * (i-OBSTACLE_X) ) + ( (j-OBSTACLE_Y) * (j-OBSTACLE_Y) ) <= OBSTACLE_R * OBSTACLE_R )
			{

				*( lbm_cell_type_t_get_cell( mesh_type , i - comm->x, j - comm->y) ) = CELL_BOUNCE_BACK;
				//for ( k = 0 ; k < DIMENSIONS ; k++)
				//	mesh[i][j][k] = 0.0;
			}
		}
	}
}

/****************************************************/
/**
 * Initialise le fluide complet avec un distribution de poiseuille correspondant un état d'écoulement
 * linéaire à l'équilibre.
 * @param mesh Le maillage à initialiser.
 * @param mesh_type La grille d'information notifiant le type des mailles.
**/
void lbm_init_global_poiseuille_profile(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type,const lbm_comm_t * comm)
{
	//vars
	int i,j,k;
	Vector v = {0.0,0.0};
	const double density = 1.0;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	//apply poiseuil for all nodes except on top/bottom border
	for ( i = 0 ; i < mesh->width ; i++)
	{
		for ( j = 0 ; j < mesh->height ; j++)
		{
			for ( k = 0 ; k < DIRECTIONS ; k++)
			{
				//compute equilibr.
				v[0] = lbm_phys_poiseuille(j + comm->y,MESH_HEIGHT);
				lbm_mesh_get_cell(mesh, i, j)[k] = lbm_phys_equilibrium_profile(v,density,k);
				//mark as standard fluid
				*( lbm_cell_type_t_get_cell( mesh_type , i, j) ) = CELL_FUILD;
				//this is a try to init the fluide with null speed except on left interface.
				//if (i > 1)
				//	lbm_mesh_get_cell(mesh, i, j)[k] = equil_weight[k];
			}
		}
	}
}

/****************************************************/
/**
 * Initialisation des conditions au bords.
 * @param mesh Le maillage à initialiser.
 * @param mesh_type La grille d'information notifiant le type des mailles.
**/
void lbm_init_border(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm)
{
	//vars
	int i,j,k;
	Vector v = {0.0,0.0};
	const double density = 1.0;

	//setup left border type
	if( comm->rank_x == 0 )
	{
		for ( j = 1 ; j < mesh->height - 1 ; j++)
			*( lbm_cell_type_t_get_cell( mesh_type , 0, j) ) = CELL_LEFT_IN;
	}

	if( comm->rank_x == comm->nb_x - 1 )
	{
		//setup right border type
		for ( j = 1 ; j < mesh->height - 1 ; j++)
			*( lbm_cell_type_t_get_cell( mesh_type , mesh->width - 1, j) ) = CELL_RIGHT_OUT;
	}

	//top
	if (comm->rank_y == 0)
		for ( i = 0 ; i < mesh->width ; i++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
			{
				//compute equilibr.
				lbm_mesh_get_cell(mesh, i, 0)[k] = lbm_phys_equilibrium_profile(v,density,k);
				//mark as bounce back
				*( lbm_cell_type_t_get_cell( mesh_type , i, 0) ) = CELL_BOUNCE_BACK;
			}

	//bottom
	if (comm->rank_y == comm->nb_y - 1)
		for ( i = 0 ; i < mesh->width ; i++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
			{
				//compute equilibr.
				lbm_mesh_get_cell(mesh, i, mesh->height - 1)[k] = lbm_phys_equilibrium_profile(v,density,k);
				//mark as bounce back
				*( lbm_cell_type_t_get_cell( mesh_type , i, mesh->height - 1) ) = CELL_BOUNCE_BACK;
			}
}

/****************************************************/
/**
 * Initialisation de l'obstacle, on bascule les types des mailles associé à CELL_BOUNCE_BACK.
 * Ici l'obstacle est un cercle de centre (OBSTACLE_X,OBSTACLE_Y) et de rayon OBSTACLE_R.
**/
#ifdef HAVE_MAGICK_WAND
void lbm_init_image_obstacle(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm,const char * fname)
{
	//vars
	int i,j;
	int k;
	int obsty;
	size_t w,h;
	PixelWand * p;

	//open wand image
	MagickBooleanType status;
	MagickWand * image = NULL;
	MagickWandGenesis();
	image = NewMagickWand();
	status = MagickReadImage(image,fname);
	if (status == MagickFalse) abort();
	p =  NewPixelWand();
	h = MagickGetImageHeight(image);
	w = MagickGetImageWidth(image);
	//printf("Obstacle size : %lu x %lu\n",w,h);

	//scale if need
	if (lbm_gbl_config.obstable_scale != 1.0)
	{
		MagickScaleImage(image,w * lbm_gbl_config.obstable_scale, h * lbm_gbl_config.obstable_scale);
		h = MagickGetImageHeight(image);
		w = MagickGetImageWidth(image);
		//printf("Obstacle size : %lu x %lu\n",w,h);
	}

	//roate
	if (lbm_gbl_config.obstable_rotate != 0.0)
	{
		PixelSetColor(p,"white");
		MagickRotateImage(image,p,lbm_gbl_config.obstable_rotate);
		h = MagickGetImageHeight(image);
		w = MagickGetImageWidth(image);
		//printf("Obstacle size after rotate : %lu x %lu\n",w,h);
	}

	obsty = (MESH_HEIGHT - h) / 2;
	//printf("Obstacle y pos : %d\n",obsty);

	//loop on nodes
	// 	#pragma omp for private (i,j)
	for ( i =  comm->x; i < mesh->width + comm->x ; i++)
	{
		for ( j =  comm->y ; j <  mesh->height + comm->y ; j++)
		{
			if ( i > OBSTACLE_X && (i-OBSTACLE_X) < w && (j-obsty) < h && j > obsty)
			{
				MagickGetImagePixelColor(image,(i-OBSTACLE_X),h - (j-obsty),p);
				if (PixelGetRed(p) < 0.8)
				{
					*( lbm_cell_type_t_get_cell( mesh_type , i - comm->x, j - comm->y) ) = CELL_BOUNCE_BACK;
					for ( k = 0 ; k < DIMENSIONS ; k++)
						lbm_mesh_get_cell(mesh,  i - comm->x, j - comm->y)[k] = equil_weight[k];
				}
			}
		}
	}
}
#endif

/****************************************************/
/**
 * Mise en place des conditions initiales.
 * @param mesh Le maillage à initialiser.
 * @param mesh_type La grille d'information notifiant le type des mailles.
 * @param comm La structure de communication pour connaitre notre position absolue dans le maillage globale.
**/
void lbm_init_mesh_state(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm)
{
	lbm_init_global_poiseuille_profile(mesh,mesh_type,comm);
	//Skip due to a bug
	//lbm_init_border(mesh,mesh_type,comm);

	#ifdef HAVE_MAGICK_WAND
		if (lbm_gbl_config.obstacle_filename == NULL)
			lbm_init_circle_obstacle(mesh,mesh_type, comm);
		else
			lbm_init_image_obstacle(mesh,mesh_type, comm,lbm_gbl_config.obstacle_filename);
	#else
		lbm_init_circle_obstacle(mesh,mesh_type, comm);
	#endif
}
