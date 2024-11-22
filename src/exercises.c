/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#include "lbm_struct.h"
#include "exercises.h"

/****************************************************/
static int gblExercice = 0;

/****************************************************/
void lbm_ex_select(int id) 
{
	int rank;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	gblExercice = id;
	if (id < 0 || id > 6)
		fatal("Invalid exercice ID !");
	if (rank == 0)
		printf("\033[32mSelect exercice %d\033[39m\n", id);
}

/****************************************************/
void lbm_comm_init_ex_select( lbm_comm_t * comm, int total_width, int total_height )
{
	switch (gblExercice) {
		case 0:
			lbm_comm_init_ex0(comm, total_width, total_height);
			break;
		case 1:
			lbm_comm_init_ex1(comm, total_width, total_height);
			break;
		case 2:
			lbm_comm_init_ex2(comm, total_width, total_height);
			break;
		case 3:
			lbm_comm_init_ex3(comm, total_width, total_height);
			break;
		case 4:
			lbm_comm_init_ex4(comm, total_width, total_height);
			break;
		case 5:
			lbm_comm_init_ex5(comm, total_width, total_height);
			break;
		case 6:
			lbm_comm_init_ex6(comm, total_width, total_height);
			break;
		default:
			fatal("Invalid exercice number !");
			break;
	}

	//check
	if (comm->nb_x == -1 || comm->nb_y == -1)
		fatal("lbm_comm_init_ex not implemented for this exercise, nb_x or nb_y is -1 !");//, gblExercice);
	if (comm->rank_x == -1 || comm->rank_y == -1)
		fatal("lbm_comm_init_ex not implemented for this exercise, rank_x or rank_y is -1 !");//, gblExercice);
	if (comm->width == -1 || comm->height == -1)
		fatal("lbm_comm_init_ex not implemented for this exercise, width or height is -1 !");//, gblExercice);
	if (comm->x == -1 || comm->y == -1)
		fatal("lbm_comm_init_ex not implemented for this exercise, x or y is -1 !");//, gblExercice);

	//check
	if (total_width % comm->nb_x != 0)
		warning("nb_x not multiple of total_width !");
	if (total_height % comm->nb_y != 0)
		warning("nb_x not multiple of total_width !");
}

/****************************************************/
void lbm_comm_release_ex_select( lbm_comm_t * comm )
{
	switch(gblExercice) {
		case 0:
			lbm_comm_release_ex0(comm);
			break;
		case 1:
			lbm_comm_release_ex0(comm);
			break;
		case 2:
			lbm_comm_release_ex0(comm);
			break;
		case 3:
			lbm_comm_release_ex0(comm);
			break;
		case 4:
			lbm_comm_release_ex4(comm);
			break;
		case 5:
			lbm_comm_release_ex5(comm);
			break;
		case 6:
			lbm_comm_release_ex6(comm);
			break;
		default:
			fatal("Invalid exercice number !");
			break;
	}
}

/****************************************************/
void lbm_comm_ghost_exchange_ex_select(lbm_comm_t * comm, lbm_mesh_t * mesh )
{
	switch(gblExercice) {
		case 0:
			lbm_comm_ghost_exchange_ex0(comm, mesh);
			break;
		case 1:
			lbm_comm_ghost_exchange_ex1(comm, mesh);
			break;
		case 2:
			lbm_comm_ghost_exchange_ex2(comm, mesh);
			break;
		case 3:
			lbm_comm_ghost_exchange_ex3(comm, mesh);
			break;
		case 4:
			lbm_comm_ghost_exchange_ex4(comm, mesh);
			break;
		case 5:
			lbm_comm_ghost_exchange_ex5(comm, mesh);
			break;
		case 6:
			lbm_comm_ghost_exchange_ex6(comm, mesh);
			break;
		default:
			fatal("Invalid exercice number !");
			break;
	}
}

/****************************************************/
void lbm_save_ex_select(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step)
{
	switch(gblExercice) {
		case 0:
			lbm_save_ex0(save_buffer, comm, mesh_to_save, mesh_type, write_step);
			break;
		case 1:
			lbm_save_ex0(save_buffer, comm, mesh_to_save, mesh_type, write_step);
			break;
		case 2:
			lbm_save_ex0(save_buffer, comm, mesh_to_save, mesh_type, write_step);
			break;
		case 3:
			lbm_save_ex0(save_buffer, comm, mesh_to_save, mesh_type, write_step);
			break;
		case 4:
			lbm_save_ex0(save_buffer, comm, mesh_to_save, mesh_type, write_step);
			break;
		case 5:
			lbm_save_ex0(save_buffer, comm, mesh_to_save, mesh_type, write_step);
			break;
		case 6:
			lbm_save_ex0(save_buffer, comm, mesh_to_save, mesh_type, write_step);
			break;
		default:
			fatal("Invalid exercice number !");
			break;
	}
}

/****************************************************/
void lbm_do_step_ex_select(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh )
{
	switch(gblExercice) {
		case 0:
			lbm_do_step_ex0(comm, mesh_type, mesh, temp_mesh );
			break;
		case 1:
			lbm_do_step_ex0(comm, mesh_type, mesh, temp_mesh );
			break;
		case 2:
			lbm_do_step_ex0(comm, mesh_type, mesh, temp_mesh );
			break;
		case 3:
			lbm_do_step_ex0(comm, mesh_type, mesh, temp_mesh );
			break;
		case 4:
			lbm_do_step_ex0(comm, mesh_type, mesh, temp_mesh );
			break;
		case 5:
			lbm_do_step_ex0(comm, mesh_type, mesh, temp_mesh );
			break;
		case 6:
			lbm_do_step_ex0(comm, mesh_type, mesh, temp_mesh );
			break;
		default:
			fatal("Invalid exercice number !");
			break;
	}
}

/****************************************************/
void lbm_save_ex0(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step)
{
	//calculate output mesh (final physics values)
	lbm_save_fill_mesh(save_buffer, mesh_to_save, mesh_type);

	//write to file
	lbm_save_write_mesh(save_buffer, comm, comm->rank_x, comm->rank_y, write_step);
}
