from functions import * 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew

red_wine = pd.read_csv('winequality-red.csv', sep = ';')
white_wine = pd.read_csv('winequality-white.csv', sep = ';')

results_red = pd.DataFrame(columns=[
    'to_reduce', 'apply_normalization', 'apply_over_sampling',
    'to_decrease_num_classes', 'epochs', 'batch_size', 'cross_val_accuracy'
])
red_wine = pd.read_csv('winequality-red.csv', sep = ';')
results_rw = run_parameter_combinations(red_wine, results_red)

results_white = pd.DataFrame(columns=[
    'to_reduce', 'apply_normalization', 'apply_over_sampling',
    'to_decrease_num_classes', 'epochs', 'batch_size', 'cross_val_accuracy'
])

white_wine = pd.read_csv('winequality-white.csv', sep = ';')
results_ww = run_parameter_combinations(white_wine, results_white)

print(results_rw)

print(results_ww)

max_row_red = results_rw.loc[results_rw['cross_val_accuracy'].idxmax()]
max_row_white = results_ww.loc[results_ww['cross_val_accuracy'].idxmax()]
print(f'Optimal parameters for Red Wine NN are \n {max_row_red}')
print(f'Optimal parameters for White Wine NN are \n {max_row_white}')

red_wine = pd.read_csv('winequality-red.csv', sep = ';')
train_neurasl_network(red_wine, to_reduce=False, 
                     apply_normalization=False, apply_over_sampling=True, 
                     to_decrease_num_classes=True, epochs=50, batch_size=32)

red_wine = pd.read_csv('winequality-red.csv', sep = ';')
train_neural_network(red_wine, to_reduce=False, apply_normalization=False, 
                     apply_over_sampling=True, to_decrease_num_classes=False, 
                     epochs=50, batch_size=32)

white_wine = pd.read_csv('winequality-white.csv', sep = ';')
train_neural_network(white_wine, to_reduce=False, apply_normalization=False, 
                     apply_over_sampling=False, to_decrease_num_classes=True, 
                     epochs=50, batch_size=32)

white_wine = pd.read_csv('winequality-white.csv', sep = ';')
train_neural_network(white_wine, to_reduce=True, 
                     apply_normalization=False, apply_over_sampling=False, 
                     to_decrease_num_classes=False, epochs=50, batch_size=32)