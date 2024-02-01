import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the output file
file_path = '/home/kelmanson/Desktop/lara/paq/centers.txt'
df = pd.read_csv(file_path, sep='\t')

# Convert timestep to time (assuming time step is 10^-3 ps)
df['Time (ps)'] = df['Timestep'] * 10**-3

# Calculate the derivative of the center coordinate with respect to time
df['Center Derivative'] = df['Center'].diff() / df['Time (ps)'].diff()
df['Center Derivative'] = df['Center Derivative'].abs()
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Time (ps)'], df['Center'], label='Center of Gaussian')
plt.xlabel('Time/ps')
plt.ylabel(r'$x/a_0$')
plt.title('Center of Gaussian over Time')
plt.legend()
plt.grid(True)

# Save the plot as SVG
output_svg_path = '/home/kelmanson/Desktop/lara/plot_1.svg'
plt.savefig(output_svg_path, format='svg')
output_svg_path

# Plotting deravative
plt.figure(figsize=(10, 6))
plt.plot(df['Time (ps)'], df['Center Derivative'], label='Velocity')
plt.xlabel('Time/ps')
plt.ylabel(r'V, $a_0$/ps')
plt.title('Velocity over Time')
plt.legend()
plt.grid(True)

# Save the plot as SVG
output_svg_path = '/home/kelmanson/Desktop/lara/plot_2.svg'
plt.savefig(output_svg_path, format='svg')
print(df['Center Derivative'].mean())


# Integration of the 'Center Derivative' over 'Time (ps)'
integral = np.trapz(df['Center Derivative'], df['Time (ps)'])

# Calculate the range of 'Time (ps)'
time_range = df['Time (ps)'].max() - df['Time (ps)'].min()

# Compute the final value by dividing the integral by the time range
#final_value = integral / time_range
#print(f'Mean velocity {final_value}')
output_svg_path
