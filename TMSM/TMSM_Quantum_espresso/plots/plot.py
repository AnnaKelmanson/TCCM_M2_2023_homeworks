import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_dos_with_modifications(filename, EFermi, output_svg):
    """
    Plots the density of states and integrated density of states from a given file with additional modifications.

    Parameters:
    filename (str): Path to the data file.
    EFermi (float): The Fermi energy value.
    output_svg (str): Output file name for the SVG.
    """
    sns.set(style='whitegrid')  # Apply Seaborn styling

    # Read data from the file
    energy, dos, int_dos = [], [], []
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip() or line.startswith('#'):
                continue  # Skip empty lines and lines starting with '#'
            parts = line.split()
            energy.append(float(parts[0]))
            dos.append(float(parts[1]))
            int_dos.append(float(parts[2]))

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Plotting DOS
    plt.subplot(1, 2, 1)
    plt.plot(energy, dos, label='Density of States')
    plt.axvline(x=EFermi, color='red', linestyle='--', label='$E_f$ = 9.683 eV')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of States $(e^-/eV)$')
    plt.title('Density of States')
    plt.legend()
    plt.grid(True)

    # Plotting Integrated DOS
    plt.subplot(1, 2, 2)
    plt.plot(energy, int_dos, label='Integrated Density of States')
    plt.axvline(x=EFermi, color='red', linestyle='--', label='$E_f$ = 9.683 eV')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Integrated Density of States $(e^-)$')
    plt.title('Integrated Density of States')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_svg, format='svg')
    plt.show()

def plot_band_structure(file_name, high_symmetry_points=None):
    # Apply Seaborn styling for consistent plot appearance
    sns.set(style='whitegrid')

    # Load data from the .gnu file
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Remove empty lines and split data
    data = [list(map(float, line.split())) for line in lines if line.strip()]

    # Convert to numpy array for easier manipulation
    data = np.array(data)
    data[:, 1:] += 0.828  # Adjust this value as needed

    # Extract k_points and eigenvalues
    k_points = data[:, 0]
    eigenvalues = data[:, 1:]

    # Increase the scale of the plot
    plt.figure(figsize=(15, 9))
    marker_size = 1
    line_width = 2

    # Plot each band separately
    num_bands = eigenvalues.shape[1]
    for band in range(num_bands):
        previous_k = k_points[0]
        start_index = 0
        for i, k in enumerate(k_points):
            if k < previous_k:  # New band starts
                plt.plot(k_points[start_index:i], eigenvalues[start_index:i, band], linestyle='-', markersize=marker_size, linewidth=line_width, color='blue')
                start_index = i
            previous_k = k
        # Plot the last segment
        plt.plot(k_points[start_index:], eigenvalues[start_index:, band],linestyle='-', markersize=marker_size, linewidth=line_width, color='blue')

    # Customize the plot
    plt.xlabel('k-points', fontsize=14)
    plt.ylabel('Energy (eV)', fontsize=14)

    # Set x-axis ticks and labels for high symmetry points
    if high_symmetry_points:
        x_ticks, x_labels = zip(*high_symmetry_points)
        plt.xticks(x_ticks, x_labels, fontsize=12)
        plt.xlim(x_ticks[0], x_ticks[-1])

    # Set y-axis limits
    plt.ylim(-11, 12)  # Adjust these limits as needed

    # Add horizontal line at 0 eV
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Add vertical lines for high symmetry points
    if high_symmetry_points:
        for point, label in high_symmetry_points:
            plt.axvline(x=point, color='gray', linestyle='--', linewidth=0.5)

    # Set plot title
    plt.title('Band Structure', fontsize=16)
    plt.savefig('WSe2-1T-b.svg', format='svg')
    # Show the plot
    plt.show()

file_name = './kz.bands.dat.gnu'
high_symmetry_points = [
    (0.0000, 'Γ'),
    (0.5774, 'M'),
    (0.9107, 'K'),
    (1.5774, 'Γ'),
    (1.6463, 'A'),
    (2.2236, 'L'),
    (2.5570, 'H'),
    (3.2236, 'A'),
]


if __name__ == '__main__':
	# Example usage
	plot_dos_with_modifications('./WSe2-1T.dos', EFermi=9.683, output_svg='WSe2-2H_plot.svg')
	plot_band_structure(file_name, high_symmetry_points)

