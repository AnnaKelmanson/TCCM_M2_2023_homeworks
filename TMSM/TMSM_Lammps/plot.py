import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

def parse_data(file_path):
    
    data_start_line = None

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.strip().startswith("Step Temp c_msd[4] v_twopoint v_fitslope"):
                data_start_line = i + 1
                break
    if data_start_line is not None:
        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=data_start_line,
                         names=["Step", "Temp", "c_msd[4]", "v_twopoint", "v_fitslope"],
                         error_bad_lines=False)
        return df
    else:
        return "Data starting line not found in the file."

def analyze_msd(filename):
    # Parse the data
    data_start_line = None
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if line.strip().startswith("Step Temp c_msd[4] v_twopoint v_fitslope"):
                data_start_line = i + 1
                break

    if data_start_line is not None:
        df = pd.read_csv(filename, delim_whitespace=True, skiprows=data_start_line,
                         names=["Step", "Temp", "c_msd[4]", "v_twopoint", "v_fitslope"],
                         error_bad_lines=False)
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce') * 0.005  # Adjust time scale if needed
        df['c_msd[4]'] = pd.to_numeric(df['c_msd[4]'], errors='coerce')
        df.dropna(subset=['Step', 'c_msd[4]'], inplace=True)

        # Linear regression
        slope, intercept, r_value, _, _ = linregress(df['Step'], df['c_msd[4]'])
        d_self = slope / 4

        # Plotting
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.regplot(x='Step', y='c_msd[4]', data=df, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})

        # Annotations
        text_annotation = f'Slope: {slope:.2f}\n' + r'$R^2$: ' + f'{r_value**2:.2f}\n' + r'$D_{\mathrm{self}}$: ' + f'{d_self:.2f}'
        ax.text(0.05, 0.95, text_annotation, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12)

        plt.xlabel('Time')
        plt.ylabel('MSD')
        plt.title('Mean Squared Displacement over Time')
        plt.savefig(f'./{filename}.png')
        plt.show()
    else:
        print("Data starting line not found in the file.")
        
if __name__ == '__main__':

	file_path = './7_log_msd.out'
	parsed_data = parse_data(file_path)

	parsed_data.head() if isinstance(parsed_data, pd.DataFrame) else parsed_data


	parsed_data['Step'] = pd.to_numeric(parsed_data['Step'], errors='coerce')
	parsed_data['c_msd[4]'] = pd.to_numeric(parsed_data['c_msd[4]'], errors='coerce')
	parsed_data['Step'] = parsed_data['Step'].apply(lambda x: x*0.005)
	parsed_data.dropna(subset=['Step', 'c_msd[4]'], inplace=True)

	slope, intercept, r_value, p_value, std_err = linregress(parsed_data['Step'], parsed_data['c_msd[4]'])

	slope

	d_self = slope / 4

	sns.set_style("whitegrid")
	plt.figure(figsize=(10, 6))
	ax = sns.regplot(x='Step', y='c_msd[4]', data=parsed_data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})

	text_annotation = f'Slope: {slope:.2f}\n' + r'$R^2$: ' + f'{r_value**2:.2f}\n' + r'$D_{\mathrm{self}}$: ' + f'{d_self:.2f}'
	ax.text(0.05, 0.95, text_annotation, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12)

	plt.xlabel('Time')
	plt.ylabel('MSD')
	plt.title('Mean Squared Displacement over Time')
	plt.savefig('./2msd_over_time.png')
	plt.show()


	analyze_msd('./7.0_log_msd.out')
	analyze_msd('./7.1_log_msd.out')
	analyze_msd('./7.2_log_msd.out')
	analyze_msd('./7.3_log_msd.out')
	analyze_msd('./7.4_log_msd.out')
	analyze_msd('./7.5_log_msd.out')
	analyze_msd('./7.6_log_msd.out')
	analyze_msd('./7.7_log_msd.out')

