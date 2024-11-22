/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/********************  HEADERS  *********************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "lbm_struct.h"

/*******************  ENUM  *********************/

typedef enum lbm_output_format_e
{
	OUT_FORMAT_GNUPLOT,
	OUT_FORMAT_OCTAVE,
	OUT_FORMAT_CHECKSUM,
	OUT_FORMAT_INFO
} lbm_output_format_t;

/*******************  FUNCTION  *********************/

void fatal(const char * message)
{
	fprintf(stderr,"FATAL ERROR : %s\n",message);
	abort();
}

/*******************  FUNCTION  *********************/

void open_data_file(lbm_data_file_t * file,const char * fname)
{
	//errors
	assert(file != NULL);
	assert(fname != NULL);

	//open file
	file->fp = fopen(fname,"r");
	if (file->fp == NULL)
	{
		perror(fname);
		abort();
	}

	//read header
	if( !fread(&file->header,sizeof(file->header),1,file->fp) )
	{
		if( ferror( file->fp ) )
		{
			perror( "fread" );
			abort();
		}

		if( feof( file->fp ) )
		{
			printf( "Error file is empty" );
			abort();
		}
	}

	//check magick
	if (file->header.magick != RESULT_MAGICK)
		fatal("Invalid file format.");

	//allocate memory
	file->entries = malloc(file->header.mesh_height * file->header.mesh_width * sizeof(lbm_file_entry_t));
}

/*******************  FUNCTION  *********************/

void close_data_file(lbm_data_file_t * file)
{
	//errors
	assert(file != NULL);
	assert(file->fp != NULL);
	assert(file->entries != NULL);

	//close
	fclose(file->fp);

	//free mem
	free(file->entries);
}

/*******************  FUNCTION  *********************/

bool read_next_frame(lbm_data_file_t * file)
{
	//vars
	size_t res;

	//errors
	assert(file != NULL);
	assert(file->fp != NULL);
	assert(file->entries != NULL);

	//load the frame
	res = fread(file->entries,sizeof(lbm_file_entry_t),file->header.mesh_height * file->header.mesh_width,file->fp);

	if (res != file->header.mesh_height * file->header.mesh_width)
	{
		if (feof(file->fp))
		{
			return false;
		} else {
			fatal("Error while reading the file.");
			return false;
		}
	} else {
		return true;
	}
}

/*******************  FUNCTION  *********************/

bool seek_to_frame(lbm_data_file_t * file,int frame)
{
	int res = fseek(file->fp,frame * sizeof(lbm_file_entry_t) * file->header.mesh_height * file->header.mesh_width,SEEK_CUR);
	return (res == 0);
}

/*******************  FUNCTION  *********************/

void print_current_frame_gnuplot(lbm_data_file_t * file)
{
	//vars
	uint32_t i,j,l;
	uint32_t line_height;
	int pos;

	//calc line_height
	line_height = file->header.mesh_height / file->header.lines;
	
	//loop on datas
	for ( i = 0 ; i < file->header.mesh_width ; i++)
	{
		for ( l = 0 ; l < file->header.lines ; l++)
		{
			for ( j = 0 ; j < line_height ; j++)
			{
					pos = line_height * i + j + l * line_height * file->header.mesh_width;
					printf("%d %d %f %f\n",i,j+l * line_height,file->entries[pos].density,file->entries[pos].v);
			}
		}
		printf("\n");
	}
	printf("\n");
}

/*******************  FUNCTION  *********************/

double checksum = 0;

void do_checksum(lbm_data_file_t * file)
{
	//vars
	uint32_t i,j,l;
	int pos;
	
	//calc line_height
	int line_height = file->header.mesh_height / file->header.lines;
	
	//loop on datas
	for ( i = 0 ; i < file->header.mesh_width ; i++)
	{
		for ( l = 0 ; l < file->header.lines ; l++)
		{
			for ( j = 0 ; j < line_height ; j++)
			{
				pos = line_height * i + j + l * line_height * file->header.mesh_width;
				checksum += file->entries[pos].density + file->entries[pos].v;
			}
		}
	}

	printf("%llX - %g\n", (unsigned long long int)checksum ,checksum);
}

/*******************  FUNCTION  *********************/
int get_frame_count(lbm_data_file_t * file)
{
	struct stat info;
	if (fstat(fileno(file->fp), &info) == 0)
		return info.st_size / (file->header.mesh_width * file->header.mesh_height * sizeof(lbm_file_entry_t));
	else
		return 0;
}

/*******************  FUNCTION  *********************/
void print_info(lbm_data_file_t * file)
{
	printf("width=%d\n",file->header.mesh_width);
	printf("height=%d\n",file->header.mesh_height);
	printf("frames=%d\n",get_frame_count(file));
}

/*******************  FUNCTION  *********************/

void print_current_frame(lbm_data_file_t * file,lbm_output_format_t format)
{
	switch(format)
	{
		case OUT_FORMAT_GNUPLOT:
			print_current_frame_gnuplot(file);
			break;
		case OUT_FORMAT_OCTAVE :
			printf("Not implemented \n");
			abort();
			break;
		case OUT_FORMAT_INFO:
			print_info(file);
			break;
		case OUT_FORMAT_CHECKSUM :
			do_checksum(file);
			break;
	}
}

/*******************  FUNCTION  *********************/

void print_data(lbm_data_file_t * file,lbm_output_format_t format,int frame)
{	
	//errors
	assert(file != NULL);
	assert(frame >= 0);

	//seek to frame
	if (seek_to_frame(file,frame) == false)
		fatal("Can't seek to the requested frame.");
	//read
	if (read_next_frame(file))
		print_current_frame(file,format);
}

/*******************  FUNCTION  *********************/

int main(int argc, char * argv[])
{
	//vars
	lbm_data_file_t file;
	lbm_output_format_t format;
	int frame = -1;
	
	//arg error
	if (argc != 4)
	{
		fprintf(stderr,"Usage : %s {--gnuplot|--octave|--checksum|--info} {file.raw} {frame_id}\n",argv[0]);
		abort();
	}

	//open
	open_data_file(&file,argv[2]);
	frame = atoi(argv[3]);

	//read args
	if (strcmp(argv[1],"--gnuplot") == 0)
		format = OUT_FORMAT_GNUPLOT;
	else if (strcmp(argv[1],"--checksum") == 0)
		format = OUT_FORMAT_CHECKSUM;
	else if (strcmp(argv[1],"--info") == 0)
		format = OUT_FORMAT_INFO;
	else		
		fatal("Invalid format option.");

	//print
	print_data(&file,format,frame);

	//close
	close_data_file(&file);

	return EXIT_SUCCESS;
}
