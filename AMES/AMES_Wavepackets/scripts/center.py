import os
import re
import numpy as np

def read_paq_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.split()
            data.append(tuple(map(float, values)))
    return data

def calculate_center(data):

    x_coordinates = np.array([d[0] for d in data])
    weights = np.array([d[3] for d in data])
    return np.average(x_coordinates, weights=weights)

def extract_timestep(filename):
    """
    Extract the timestep from the filename.
    Assumes filename format 'paq<number>.txt'
    """
    match = re.search(r'paq(\d+)', filename)
    return int(match.group(1)) if match else None

def process_files(directory):
    file_pattern = re.compile(r'paq\d+')

    # Sorting the files to ensure correct temporal order
    files = sorted([f for f in os.listdir(directory) if file_pattern.match(f)], 
                   key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    results = []

    for file in files:
        timestep = extract_timestep(file)
        data = read_paq_file(os.path.join(directory, file))
        center = calculate_center(data)
        results.append((timestep, center))

    return results

def write_results_to_file(results, output_filename):
    with open(output_filename, 'w') as file:
        file.write("Timestep\tCenter\n")
        for timestep, center in results:
            file.write(f"{timestep}\t{center}\n")

# Example usage (update paths as needed)
directory = '/home/kelmanson/Desktop/lara_again/plot/'  # Update this path
output_filename = '/home/kelmanson/Desktop/lara_again/centers.txt'  # Update this path

results = process_files(directory)
write_results_to_file(results, output_filename)



