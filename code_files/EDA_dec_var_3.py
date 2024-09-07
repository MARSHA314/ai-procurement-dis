import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load dataframe
df = pd.read_csv('data/Data_anonym_clean.csv')

# Create the histogram for the 'CONTRACT WEIGHT' column
df['CONTRACT WEIGHT'].plot(kind='hist', bins=100, edgecolor='black')

plt.title('Histogram of Contract Weights')
plt.xlabel('Contract Weight')
plt.ylabel('Frequency')

file_path = 'graphs/GRAPH_histogram_contract_weight.jpg'
plt.savefig(file_path, format='jpg', dpi=300)

plt.show()

# Get summary statistics of 'CONTRACT WEIGHT'
df_describe = df['CONTRACT WEIGHT'].describe()

measure_descriptions = {
    'Measure': [
        'Count of values',
        'Mean of values',
        'Standard deviation of values',
        'Minimum value',
        '25th percentile (1st quartile)',
        'Median (50th percentile)',
        '75th percentile (3rd quartile)',
        'Maximum value'
    ]
}

descriptions_df = pd.DataFrame(measure_descriptions, index=df_describe.index)
result_df = pd.concat([descriptions_df, df_describe], axis=1)
csv_file_path = 'graphs/Stat_description_CW.csv'
result_df.to_csv(csv_file_path, index=False)

# Normalise the contract weight by applying the Box-Cox transform
transformed_data, lambda_value = stats.boxcox(df['CONTRACT WEIGHT'])

# Add the transformed data as a new column in the DataFrame
df['STD CONTRACT WEIGHT'] = transformed_data

print("Lambda value:", lambda_value)
#λ = -1: Inverse transformation (1/x).
#λ = -0.5: Reciprocal square root transformation (1/√x).
#λ = 0: Log transformation (log(x)).
#λ = 0.5: Square root transformation (√x).
#λ = 1: No transformation.

# Create the histogram for the normalised 'weight' column
df['STD CONTRACT WEIGHT'].plot(kind='hist', bins=100, edgecolor='black')

plt.title('Histogram of Normalised Contract Weights')
plt.xlabel('Normalised Contract Weight, lambda=0.05')
plt.ylabel('Frequency')

file_path = 'graphs/GRAPH_histogram_contract_weight_normalised.jpg'
plt.savefig(file_path, format='jpg', dpi=300)

plt.show()

# Get summary statistics od the normalised contract weight
df_describe = df['STD CONTRACT WEIGHT'].describe()

measure_descriptions = {
    'Measure': [
        'Count of values',
        'Mean of values',
        'Standard deviation of values',
        'Minimum value',
        '25th percentile (1st quartile)',
        'Median (50th percentile)',
        '75th percentile (3rd quartile)',
        'Maximum value'
    ]
}

descriptions_df = pd.DataFrame(measure_descriptions, index=df_describe.index)
result_df = pd.concat([descriptions_df, df_describe], axis=1)
csv_file_path = 'graphs/Stat_description_CW_normalised.csv'
result_df.to_csv(csv_file_path, index=False)

# Define conditions and corresponding choices for classification and create CONTRACT CLASS column
conditions = [
    df['STD CONTRACT WEIGHT'] >= 0.68,
    df['STD CONTRACT WEIGHT'] <= -0.68
]

choices = [
    'High',
    'Low'
]

df['CONTRACT CLASS'] = np.select(conditions, choices, default='Standard')

# Create subplots of contract class
classes = df['CONTRACT CLASS'].unique()
order = ['Low', 'Standard', 'High']
fig, axs = plt.subplots(1, len(classes), figsize=(15, 5), sharey=True)

# Plot each class in a separate subplot
for ax, cls in zip(axs,  order):
    subset = df[df['CONTRACT CLASS'] == cls]
    ax.hist(subset['CONTRACT QTY'], bins=20,  alpha=0.7)
    count = subset['CONTRACT QTY'].count()
    ax.set_title(f'Class {cls}\nCount: {count}')
    ax.set_xlim(0, 1000000)
    ax.set_xlabel('Contract QTY')

fig.suptitle('Histogram by Class Variable')
fig.supxlabel('Contract QTY')
fig.supylabel('Frequency')

file_path = 'graphs/GRAPH_qty_histogram_by_class_subplot.jpg'
plt.savefig(file_path, format='jpg', dpi=300)

plt.show()

# Output in file
file_path = 'data/Data_anonym_w_class.csv'
df.to_csv(file_path, index=False)
print(df.shape)