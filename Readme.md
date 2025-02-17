# CUDA Implementation of the 2D Lattice-Boltzmann Method on GPUs

## Overview
This project presents a CUDA-based implementation of the 2D Lattice-Boltzmann Method (LBM) optimized for NVIDIA GPUs. Leveraging CUDA capabilities, this implementation ensures efficient parallel processing, making it suitable for high-performance simulations.

### Authors
- **Romain Alves**  (romalves70@gmail.com)
- **Thilen Savignon**
- **Julien Houny**

---

## Requirements
- **CUDA 12.6+**
- **g++ 11+**
- **Python 3** with the following libraries:
  - `subprocess`
  - `matplotlib`
  - `tqdm`
  - `re`
- **gnuplot** for visualization

Ensure that the latest CUDA drivers are installed along with a compatible Python environment.

---

## Compilation
A `Makefile` is provided for compiling the project. To compile, run:
```sh
make
```
in the project directory.

---

## Execution
The program can be executed with:
```sh
./main --args=value
```
For a full list of command-line arguments, use:
```sh
./main -h
```
or
```sh
./main --help
```

---

## Shell Scripts
Several shell scripts are provided to streamline execution:

### 1. Running an Example Simulation
```sh
$ ./Exec.sh <nx> <ny> [iter] [shared]
```
- `nx`, `ny`: Grid dimensions (must be multiples of 32 and > 32)
- `iter`: Number of iterations (optional)
- `shared`: Use "s" to enable shared memory optimization (optional, highly recommended)

### 2. Running with a Configuration File
```sh
$ ./Exec_config.sh <config_file> [iter] [shared]
```
- `config_file`: Path to a TXT configuration file specifying dimensions and matrix configuration.
- `iter`, `shared`: As described above.

### 3. Generating a GIF Output
```sh
$ ./GiFi.sh <nx> <ny> [shared]
```
- `nx`, `ny`, `shared`: As described above.

---

## Configuration File Format
Configuration files are TXT files structured as follows:
```
<size-x> <size-y>
m_0_0 ... m_size-x_0
...
m_0_size-y ... m_size-x_size-y
```
Where `m_x_y` represents cell types:
- `0`: Fluid cell
- `1`: Wall cell
- `2`: Driving cell

Example:
```
64 64
0 1 0 1 ...
1 0 1 0 ...
...
```

---

## Shared Memory Optimization
To utilize shared memory for better performance, include the "s" argument when executing the scripts. Note that the program supports grid sizes in multiples of CUDA block sizes (32 without shared memory, 16 with shared memory). Non-compliant sizes will result in undefined behavior.

---

## Benchmarking
A `Benchmark.py` script is provided to measure GPU efficiency, execution times, and generate performance graphs.

---

## Final Notes
- Ensure input dimensions are multiples of 32 and > 32.
- Use the "s" argument for shared memory.
- Utilize `Benchmark.py` for performance evaluation.

This CUDA implementation facilitates efficient LBM simulations with comprehensive configurability and visualization tools.

