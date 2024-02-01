import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = './parsed_results.txt'

df = pd.read_csv(file_path, delimiter=',')


rename_dict = {
    "jacobi-serial-fortran": "Serial",
    "jacobi-mpi-block-fortran": "Blocking MPI",
    "jacobi-mpi-nonblock-fortran": "Non-blocking MPI",
    "jacobi-mpi-sendrecv-fortran": "MPI SENDRECV"
}

df['Executable'] = df['Executable'].replace(rename_dict)

sns.set_theme()
for grid_size in [100, 200, 300]:
    df_filtered = df[df['Grid Size'] == grid_size]
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='MPI Ranks', y='Execution Time', hue='Executable', marker='o')

    plt.title(f"Execution Time vs MPI Ranks for Grid Size {grid_size}")
    plt.xlabel("MPI ranks")
    plt.ylabel("time/seconds")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"execution_time_mpi_ranks_grid_{grid_size}.svg", format='svg')
    #plt.show()

