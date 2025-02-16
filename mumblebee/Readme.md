# GP-GPU Project: Lattice-Blotzmann Method CUDA implementation

Authors: ALVES Romain, SAVIGNON Thilen, HOUNY Julien

## Compilation

Tested with CUDA 12.6+ and g++ 11+

A Makefile is provided for compilation, simply run
```
make
```
in the project directory.

## Running the project

The program can be executed using
```
./main --args=value
```

The list of arguments can be displayed by running
```
./main -h
```
```
./main --help
```

Additionally, some scripts are provided to assist in running the program. These scripts require gnuplot installed.

```
./Exec.sh
```
Runs the program and generates a PNG image with the output.
```
./Exec_config.sh
```
Is similar to the previous script but takes a TXT config file as an input. See Configuration File section for the file format.
```
./GiFi.sh
```
Creates a 25 frames GIF file from the simulation, taking only a size as an input.

## Configuration File

The main program and the Exec_config.sh script accept config files in TXT format with the following structure:
```
size-x size-y
  m_0_0  . . . . m_size-x_0
     .   .            .
     .     .          . 
     .       .        .
     .         .      .
m_0_size-y     m_size-x_size-y
```
Where m_x_y is the code for the (x, y) cell with the following value:
- 0: Fluid cell
- 1: Wall cell
- 2: Driving cell

## Known bugs

As of now, the program only supports grid sizes in multiple of the CUDA block size (32 without shared memory, 16 with shared memory). Any non-compliant size will result in undefined behaviour.