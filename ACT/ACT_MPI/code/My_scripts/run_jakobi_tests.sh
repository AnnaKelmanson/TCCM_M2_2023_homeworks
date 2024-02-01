#!/bin/bash

# Load the necessary MPI module
module load autoload openmpi

# Check if the MPI module loaded successfully
if [ $? -ne 0 ]; then
    echo "Failed to load autoload openmpi module"
    exit 1
fi
echo "autoload openmpi module loaded successfully"

# Compile the serial Fortran code using gfortran
gfortran -o jacobi-serial-fortran jacobi-serial.f90
if [ $? -ne 0 ]; then
    echo "Compilation of jacobi-serial.f90 failed"
    exit 1
fi
echo "Compilation of jacobi-serial.f90 completed"

# Compile Fortran MPI versions using mpif90
mpif90 -o jacobi-mpi-block-fortran jacobi-mpi-block.f90
mpif90 -o jacobi-mpi-nonblock-fortran jacobi-mpi-nonblock.f90
mpif90 -o jacobi-mpi-sendrecv-fortran jacobi-mpi-sendrecv.f90

# Define the executable names
serial_executable="jacobi-serial-fortran"
blocking_mpi_executable="jacobi-mpi-block-fortran"
non_blocking_mpi_executable="jacobi-mpi-nonblock-fortran"
sendrecv_mpi_executable="jacobi-mpi-sendrecv-fortran"

# Define an array of grid sizes
grid_sizes=(100 200 300)

# Define an array of MPI ranks
mpi_ranks=(1 2 4 6 8 12 16 20 24 28 32)

# Output file
output_file="output.txt"

# Initialize the output file with headers
echo "Executable,Grid Size,MPI Ranks,Execution Time" > $output_file

# Function to run a specific executable with all combinations of grid sizes and MPI ranks
run_tests() {
    local executable=$1
    local is_serial=$2
    for grid in "${grid_sizes[@]}"; do
        for ranks in "${mpi_ranks[@]}"; do
            echo "Running $executable with grid size $grid and $ranks MPI ranks"
            if [ "$is_serial" == "yes" ]; then
                exec_time=$( (time srun -n$ranks ./$executable $grid $grid) 2>&1 )
            else
                exec_time=$(srun -n$ranks ./$executable $grid $grid 2>&1)
            fi
            echo "$executable,$grid,$ranks,$exec_time" >> $output_file
        done
    done
}

# Run tests for all versions
run_tests $serial_executable "yes"
run_tests $blocking_mpi_executable "no"
run_tests $non_blocking_mpi_executable "no"
run_tests $sendrecv_mpi_executable "no"

echo "Results saved in $output_file"

