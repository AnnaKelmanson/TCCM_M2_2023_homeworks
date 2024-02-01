import numpy as np
import matplotlib.pyplot as plt
import os
import re
import random

def select_random_files(directory, max_timestep, num_files=8):
    """
    Selects num_files random files from the directory with timestep less than max_timestep.
    """
    file_pattern = re.compile(r'paq(\d+)')
    all_files = [f for f in os.listdir(directory) if file_pattern.match(f)]
    filtered_files = [f for f in all_files if int(file_pattern.match(f).group(1)) < max_timestep]

    # Ensuring we have enough files to select from
    if len(filtered_files) < num_files:
        raise ValueError("Not enough files to select from.")

    selected_files = random.sample(filtered_files, num_files)
    selected_files.sort(key=lambda x: int(file_pattern.match(x).group(1)))  # Sorting by timestep
    return selected_files
def extract_timestep(filename):
    """
    Extract the timestep from the filename.
    Assumes filename format 'paq<number>.txt'
    """
    match = re.search(r'paq(\d+)', filename)
    return int(match.group(1)) if match else None

def plot_gaussian(file_path):
    """
    Plot a Gaussian from a given file.
    """
    data = np.loadtxt(file_path, usecols=[0, 3])  # x-coordinate and modulus square
    plt.plot(data[:, 0], data[:, 1], label=f'Timestep {extract_timestep(file_path)}')

# Example usage
directory = '/home/kelmanson/Desktop/lara/paq/'  # Update this path
selected_files = select_random_files(directory, max_timestep=5000)

plt.figure(figsize=(12, 8))

for file in selected_files:
    plot_gaussian(os.path.join(directory, file))

plt.xlabel(r'$x/a_0$')
plt.ylabel('|Ψ(x)|²')
plt.title('Wavepacket Squared Amplitude')
plt.legend()
plt.grid(True)

# Save the plot
output_plot_path = '/home/kelmanson/Desktop/lara/gaussian_plot.svg'  # Update this path
plt.savefig(output_plot_path, format='svg')

output_plot_path

