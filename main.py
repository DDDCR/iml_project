import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

sns.set_style('whitegrid')

columns_to_plot = ['MW', 'NumOfAtoms', 'NumOfC', 'NumOfO', 'NumOfN', 'NumHBondDonors', 'NumOfConf', 'NumOfConfUsed']

num_cols = len(columns_to_plot)

plt.figure(figsize=(14, 10))

for idx, col in enumerate(columns_to_plot):
    plt.subplot((num_cols + 2) // 3, 3, idx + 1)  # Arrange subplots dynamically
    sns.violinplot(data=train_data[col], orient='h', inner="quartile", color='lightcoral')
    plt.title(f'{col} Distribution', fontsize=10)
    plt.xlabel('Values')
    plt.ylabel('Density')

plt.tight_layout()

plt.show()

plt.figure(figsize=(12, 7))
train_data['parentspecies'].value_counts().plot(
    kind='bar', 
    color='mediumpurple', 
    edgecolor='black'
)

plt.title('Distribution of Parent Species', fontsize=14)
plt.xlabel('Parent Species', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=30, ha='right') 

plt.tight_layout()

plt.show()
