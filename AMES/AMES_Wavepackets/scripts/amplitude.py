import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the output file
file_path = '/home/kelmanson/Desktop/lara/2i/paq0.txt'
df = pd.read_csv(file_path, sep=' ')

# Calculate the derivative of the center coordinate with respect to time
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['x'], df['wpms'], label='Wavepacket modulus square')
plt.xlabel(r'x/$a_0$')
plt.ylabel('|Ψ(x)|²')
plt.title('Wavepacket Squared Amplitude')
plt.legend()
plt.grid(True)

# Save the plot as SVG
output_svg_path = '/home/kelmanson/Desktop/lara/2i/plot_1.svg'
plt.savefig(output_svg_path, format='svg')
output_svg_path

