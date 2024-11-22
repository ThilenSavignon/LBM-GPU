/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#ifndef LBM_CONFIG_H
#define LBM_CONFIG_H

/****************************************************/

//number of space dimentions to consider
#define DIMENSIONS 2
#define DIRECTIONS 9
//mesh discretisation
#define MESH_WIDTH (lbm_gbl_config.width)
#define MESH_HEIGHT (lbm_gbl_config.height)
//obstable parameter
#define OBSTACLE_R (lbm_gbl_config.obstacle_r)
#define OBSTACLE_X (lbm_gbl_config.obstacle_x)
#define OBSTACLE_Y (lbm_gbl_config.obstacle_y)
//time discretisation
#define ITERATIONS (lbm_gbl_config.iterations)
//initial conditions
//velocity of fluide on left input interface
#define INFLOW_MAX_VELOCITY (lbm_gbl_config.inflow_max_velocity)
//fluid parameters
#define REYNOLDS (lbm_gbl_config.reynolds)
#define KINETIC_VISCOSITY (lbm_gbl_config.kinetic_viscosity)
#define RELAX_PARAMETER (lbm_gbl_config.relax_parameter)
//result filename
#define RESULT_FILENAME (lbm_gbl_config.output_filename)
#define RESULT_MAGICK 0x12345
#define WRITE_BUFFER_ENTRIES 4096
#define WRITE_STEP_INTERVAL (lbm_gbl_config.write_interval)

/****************************************************/
/**
 * Structure de configuration du problème à résoudre.
**/
typedef struct lbm_config_s
{
	//discretisation
	int iterations;
	int width;
	int height;
	//obstacle
	double obstacle_r;
	double obstacle_x;
	double obstacle_y;
	//flow parameters
	double inflow_max_velocity;
	double reynolds;
	//derived flow parameters
	double kinetic_viscosity;
	double relax_parameter;
	//results
	const char * output_filename;
	int write_interval;
	//obstable
	const char * obstacle_filename;
	double obstable_scale;
	double obstable_rotate;
} lbm_config_t;

/****************************************************/
void lbm_config_init(const char * filename);
void lbm_config_drived_parameters(void);
void lbm_config_cleanup(void);
void lbm_config_print(void);
void lbm_config_set_default(void);

/****************************************************/
/**
 * Configuration accessible sous le forme d'une variable globale.
 * A utiliser comme une constante à part pour le chargement initiale.
**/
extern lbm_config_t lbm_gbl_config;

#endif //LBM_CONFIG_H
