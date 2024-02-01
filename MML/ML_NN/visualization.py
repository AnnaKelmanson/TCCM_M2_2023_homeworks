from functions import *

import pandas as pd  
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew
import seaborn as sns

red_wine = pd.read_csv('winequality-red.csv', sep = ';')
white_wine = pd.read_csv('winequality-white.csv', sep = ';')

X_red, Y_red = splitting_into_X_Y(red_wine) 
X_white, Y_white = splitting_into_X_Y(white_wine)

red_wine.describe()  

red_wine.isnull().sum()   

Y_red['quality'].plot(kind='hist', bins=20, title='quality')
plt.gca().spines[['top', 'right',]].set_visible(False)

vs_plots(red_wine)

plot_correlation(red_wine)  

boxplots(red_wine)

sns.distplot(red_wine['alcohol'])

skew(red_wine['alcohol'])

fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()
for col, value in red_wine.items():
    sns.distplot(value, color='r', ax=ax[index]) 
    index += 1
    
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)