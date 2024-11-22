/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "lbm_config.h"

/****************************************************/

lbm_config_t lbm_gbl_config;

/****************************************************/
/**
 * Application des valeurs par defaut au cas ou l'utilisateur en définirait pas tout dans
 * le fichie de configuration.
**/
void lbm_config_set_default(void)
{
	//directisation.
	lbm_gbl_config.iterations = 10000;
	lbm_gbl_config.width = 800;
	lbm_gbl_config.height = 100;
	//obstacle
	lbm_gbl_config.obstacle_r = 0.0;
	lbm_gbl_config.obstacle_x = 0.0;
	lbm_gbl_config.obstacle_y = 0.0;
	//flow
	lbm_gbl_config.inflow_max_velocity = 0.1;
	lbm_gbl_config.reynolds = 100;
	//result output file
	lbm_gbl_config.output_filename = NULL;
	lbm_gbl_config.write_interval = 50;
	//obstacle
	lbm_gbl_config.obstacle_filename = NULL;
	lbm_gbl_config.obstable_scale = 1.0;
	lbm_gbl_config.obstable_rotate = 0.0;
}

/****************************************************/
/**
 * Calcule des paramètres dérivés.
**/
void lbm_config_drived_parameters(void)
{
	//derived parameter
	lbm_gbl_config.kinetic_viscosity = (lbm_gbl_config.inflow_max_velocity * 2.0 * lbm_gbl_config.obstacle_r / lbm_gbl_config.reynolds);
	lbm_gbl_config.relax_parameter = 1.0 / (3.0 * lbm_gbl_config.kinetic_viscosity + 1.0/2.0);
}

/****************************************************/
/**
 * Chargement de la config depuis le fichier.
**/
void lbm_config_init(const char * filename)
{
	//vars
	FILE * fp;
	char buffer[1024];
	char buffer2[1024];
	int intValue;
	double doubleValue;
	int line = 0;

	//open the config file
	fp = fopen(filename,"r");
	if (fp == NULL)
	{
		perror(filename);
		abort();
	}

	//load default values
	lbm_config_set_default();

	//loop on lines
	while (fgets(buffer,1024,fp) != NULL)
	{
		line++;
		if (buffer[0] == '#')
		{
			//comment, nothing to do
		} else if (sscanf(buffer,"iterations = %d\n",&intValue) == 1) {
			 lbm_gbl_config.iterations = intValue;
		} else if (sscanf(buffer,"width = %d\n",&intValue) == 1) {
			 lbm_gbl_config.width = intValue;
			 if (lbm_gbl_config.obstacle_x == 0.0)
				lbm_gbl_config.obstacle_x = (lbm_gbl_config.width / 5.0 + 1.0);
		} else if (sscanf(buffer,"height = %d\n",&intValue) == 1) {
			lbm_gbl_config.height = intValue;
			if (lbm_gbl_config.obstacle_r == 0.0)
				lbm_gbl_config.obstacle_r = (lbm_gbl_config.height / 10.0 + 1.0);
			if (lbm_gbl_config.obstacle_y == 0.0)
				lbm_gbl_config.obstacle_y = (lbm_gbl_config.height / 2.0 + 3.0);
		} else if (sscanf(buffer,"obstacle_r = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.obstacle_r = doubleValue;
		} else if (sscanf(buffer,"obstacle_x = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.obstacle_x = doubleValue;
		} else if (sscanf(buffer,"obstacle_y = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.obstacle_y = doubleValue;
		} else if (sscanf(buffer,"inflow_max_velocity = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.inflow_max_velocity = doubleValue;
		} else if (sscanf(buffer,"reynolds = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.reynolds = doubleValue;
		} else if (sscanf(buffer,"kinetic_viscosity = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.kinetic_viscosity = doubleValue;
		} else if (sscanf(buffer,"relax_parameter = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.relax_parameter = doubleValue;
		} else if (sscanf(buffer,"write_interval = %d\n",&intValue) == 1) {
			 lbm_gbl_config.write_interval = intValue;
		} else if (sscanf(buffer,"output_filename = %s\n",buffer2) == 1) {
			 lbm_gbl_config.output_filename = strdup(buffer2);
		} else if (sscanf(buffer,"obstacle_filename = %s\n",buffer2) == 1) {
			 lbm_gbl_config.obstacle_filename = strdup(buffer2);
		} else if (sscanf(buffer,"obstacle_scale = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.obstable_scale = doubleValue;
		} else if (sscanf(buffer,"obstacle_rotate = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.obstable_rotate = doubleValue;
		} else {
			fprintf(stderr,"Invalid config option line %d : %s\n",line,buffer);
			abort();
		}
	}

	//check error
	if (!feof(fp))
	{
		perror(filename);
		abort();
	}

	lbm_config_drived_parameters();
}

/****************************************************/
/**
 * Nettotage de la mémoire dynamique de la config.
**/
void lbm_config_cleanup(void)
{
	free((void*)lbm_gbl_config.output_filename);
}

/****************************************************/
/**
 * Affichage de la config.
**/
void lbm_config_print(void)
{
	printf("=================== CONFIG ===================\n");
	//discretisation
	printf("%-20s = %d\n","iterations",lbm_gbl_config.iterations);
	printf("%-20s = %d\n","width",lbm_gbl_config.width);
	printf("%-20s = %d\n","height",lbm_gbl_config.height);
	//obstacle
	printf("%-20s = %lf\n","obstacle_r",lbm_gbl_config.obstacle_r);
	printf("%-20s = %lf\n","obstacle_x",lbm_gbl_config.obstacle_x);
	printf("%-20s = %lf\n","obstacle_y",lbm_gbl_config.obstacle_y);
	//flow parameters
	printf("%-20s = %lf\n","reynolds",lbm_gbl_config.reynolds);
	printf("%-20s = %lf\n","inflow_max_velocity",lbm_gbl_config.inflow_max_velocity);
	printf("%-20s = %lf\n","inflow_max_velocity",lbm_gbl_config.inflow_max_velocity);
	//results
	printf("%-20s = %s\n","output_filename",lbm_gbl_config.output_filename);
	printf("%-20s = %d\n","write_interval",lbm_gbl_config.write_interval);
	//obstacle
	printf("%-20s = %s\n","obstacle_filename",lbm_gbl_config.obstacle_filename);
	printf("%-20s = %lf\n","obstable_scale",lbm_gbl_config.obstable_scale);
	printf("%-20s = %lf\n","obstable_rotate",lbm_gbl_config.obstable_rotate);
	printf("------------ Derived parameters --------------\n");
	printf("%-20s = %lf\n","kinetic_viscosity",lbm_gbl_config.kinetic_viscosity);
	printf("%-20s = %lf\n","relax_parameter",lbm_gbl_config.relax_parameter);
	printf("==============================================\n");
}
