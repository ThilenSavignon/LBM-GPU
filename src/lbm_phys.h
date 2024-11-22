/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#ifndef LBM_PHYS_H
#define LBM_PHYS_H

/****************************************************/
#include "lbm_struct.h"
#include "lbm_comm.h"

/****************************************************/
extern const int opposite_of[DIRECTIONS];
extern const double equil_weight[DIRECTIONS];
extern const Vector direction_matrix[DIRECTIONS];

/****************************************************/
//helper
double lbm_phys_vect_norme_2(const Vector vect1,const Vector vect2);
double lbm_phys_cell_density(const lbm_mesh_cell_t cell);
void lbm_phys_cell_velocity(Vector v,const lbm_mesh_cell_t cell,double cell_density);
double lbm_phys_poiseuille(int i,int size);

/****************************************************/
//collistion
double lbm_phys_equilibrium_profile(Vector velocity,double density,int direction);
void lbm_phys_cell_lbm_phys_collision(lbm_mesh_cell_t cell_out,const lbm_mesh_cell_t cell_in);

/****************************************************/
//limit conditions
void lbm_phys_bounce_back(lbm_mesh_cell_t cell);
void lbm_phys_inflow_zou_he_poiseuille_distr( const lbm_mesh_t * mesh, lbm_mesh_cell_t cell,int id_y);
void lbm_phys_outflow_zou_he_const_density(lbm_mesh_cell_t mesh);

/****************************************************/
//main functions
void lbm_phys_special_cells(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm);
void lbm_phys_special_cells_inner(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm);
void lbm_phys_special_cells_border(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm);
void lbm_phys_collision(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in);
void lbm_phys_collision_inner(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in);
void lbm_phys_collision_border(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in);
void lbm_phys_propagation(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in);
void lbm_phys_propagation_border(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in);
void lbm_phys_propagation_inner(lbm_mesh_t * mesh_out,const lbm_mesh_t * mesh_in);

#endif
