import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset to see the contents and structure
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Filtering out non-numeric and the 'ID' column
plot_data = data.select_dtypes(include=[float, int]).drop(columns=['ID'])

# Plotting
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
axes = axes.ravel()

for i, col in enumerate(plot_data.columns[1:]):  # Skip 'log_pSat_Pa' since it's the x-axis
    axes[i].scatter(plot_data['log_pSat_Pa'], plot_data[col])
    axes[i].set_title(f'log_pSat_Pa vs {col}')
    axes[i].set_xlabel('log_pSat_Pa')
    axes[i].set_ylabel(col)

# Hide unused subplots if there are any
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.show()
