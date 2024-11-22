/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "lbm_comm.h"

#define XID 1
#define YID 0

/****************************************************/
/**
 * Calcule le PGCD de deux nombre pour trouver un multitple
 * de la taille du maillage pour la découpe des domaines MPI.
**/
int lbm_helper_pgcd(int a, int b)
{
	int c;
	while(b!=0)
	{
		c = a % b;
		a = b;
		b = c;
	}
	return a;
}

/****************************************************/
/**
 * Affiche la configuation du lbm_comm pour un rank donné
 * @param comm Configuration à afficher
**/
void  lbm_comm_print( lbm_comm_t *comm )
{
	int rank ;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	printf( " RANK %d ( POSITION %d %d ) (WH %d %d ) \n", 
		rank,
		comm->x,
		comm->y,
		comm->width,
		comm->height);
}
