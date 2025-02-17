# LBM-GPU

## CUDA Implementation of the 2D Squared Lattice-Boltzmann Method

This project provides a CUDA-based implementation of the 2D squared Lattice-Boltzmann Method (LBM) designed to run efficiently on NVIDIA GPUs.

---

## Requirements
- **CUDA (latest version)**
- **Python 3**
  - Required Python libraries:
    - `subprocess`
    - `matplotlib`
    - `tqdm`
    - `re`

Ensure that you have the latest CUDA drivers installed and a functional Python 3 environment with the specified libraries.

---

## Shell Scripts

There are three main shell scripts included in the project, each with specific functionalities.

### 1. Script for Displaying an Example Simulation
**Usage:**
```sh
$ ./Exec.sh <nx> <ny> [iter] [shared]
```
- **nx**: Number of columns (must be a multiple of 32 and greater than 32)
- **ny**: Number of rows (must be a multiple of 32 and greater than 32)
- **iter**: Number of iterations (optional)
- **shared**: Use "s" to enable shared memory optimizations (optional but highly recommended for performance)

**Note:** This script displays an example simulation with the specified dimensions.

---

### 2. Script for Running with a Configuration File
**Usage:**
```sh
$ ./Exec_config.sh <config_file> [iter] [shared]
```
- **config_file**: Path to a configuration file containing the number of columns and rows on the first line, followed by the matrix configuration.
- **iter**: Number of iterations (optional)
- **shared**: Use "s" to enable shared memory optimizations (optional but highly recommended for performance)

**Note:** The configuration file should follow the format provided in the `example_config` file.

---

### 3. Script for Generating a GIF Output
**Usage:**
```sh
$ ./GiFi.sh <nx> <ny> [shared]
```
- **nx**: Number of columns (must be a multiple of 32 and greater than 32)
- **ny**: Number of rows (must be a multiple of 32 and greater than 32)
- **shared**: Use "s" to enable shared memory optimizations (optional but highly recommended for performance)

**Note:** This script generates a GIF of the simulation. Be aware that this process may take a considerable amount of time.

---

## Important Note on Shared Memory Usage
To enable shared memory optimization, you **must** pass the argument "s" when executing any of the shell scripts. This ensures that the CUDA kernels utilize shared memory, resulting in significantly improved performance.

---

## Benchmarking Script
A Python script named `Benchmark.py` is provided to calculate the efficiency of the GPU implementation.

**Required Python libraries:**
- `subprocess`
- `matplotlib.pyplot`
- `tqdm`
- `re`

The benchmarking script is designed to measure execution times and generate efficiency graphs, making it easier to analyze the performance of the LBM implementation across different configurations.

---

## Example Configuration File (`example_config`)
```
64 64
0 1 0 1 0 1 0 1 ...
1 0 1 0 1 0 1 0 ...
...
```
The first line specifies the dimensions (nx and ny), followed by the matrix configuration.

---

## Final Notes
- Ensure that all input dimensions (nx and ny) are multiples of 32 and greater than 32.
- Always pass "s" as an argument in the shell scripts to utilize shared memory for better performance.
- Use the provided `Benchmark.py` script to evaluate the performance of your simulations.

---

This CUDA implementation leverages the power of GPUs to efficiently simulate the Lattice-Boltzmann Method in 2D, offering configurability, performance optimizations, and visualization options through shell scripts and Python benchmarking tools.

---

### Author
Romain Alves

Do not hesitate to contact me if you have any problems: romalves70@gmail.com