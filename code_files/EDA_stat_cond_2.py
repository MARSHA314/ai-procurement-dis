import pandas as pd
import numpy as np
import io

# Load dataframe
df = pd.read_csv('data/Data_anonym.csv')

# Get information about data types and missing values
file_path = 'graphs/Stat_info_dataset.txt'

buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

with open(file_path, 'w') as file: 
    file.write(info_str)

# Group by the 'year' column and count the records
counts_per_year = df.groupby('YEAR').size().reset_index(name='counts')
print(counts_per_year)

# Create contract weight measure as contract qty over std batch 
df['CONTRACT WEIGHT']=np.where(df['STD BATCH'] == 0, 1, df['CONTRACT QTY']/df['STD BATCH'])

# Select only measure columns
columns_to_keep = [
   'YEAR', 'CONTRACT QTY', 'PRICE', 'CD_DOWN_INTERVAL', 'VL_BATCH', 'VL_MPVS', 'VL_MIN_ORDER_QTY', 
    'BATCH', 'MPVS', 'FY', 'FY_LY', 'STD BATCH','CONTRACT WEIGHT', 'FY WOW','VS CURRENT', 'VS PREV', 
    'AVG_COVERAGE_ONLY', 'AVG_COVERAGE_ONLY_Q2', 'TOTAL_DEMAND_ONLY', 
    'TOTAL_DEMAND_ONLY_Q2', 'AVG_DEMAND_ONLY', 'AVG_DEMAND_ONLY_Q2', 
    'TOTAL_PREBUILD', 'TOTAL_PREBUILD_Q2', 'AVG_PREBUILD', 'AVG_PREBUILD_Q2', 
    'AVG_INVENTORY_ONLY', 'AVG_INVENTORY_ONLY_Q2', 'AVG_SAFETY', 'AVG_SAFETY_Q2', 
    'AVG_SUPPLY', 'AVG_SUPPLY_Q2'
]
df_measure = df.loc[:, columns_to_keep] 


# Get summary statistics and save them in file
df_describe = df_measure.describe()

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

csv_file_path = 'graphs/Stat_description_dataset.csv'
result_df.to_csv(csv_file_path, index=False)

# Conditions to remove outliers based on the summary statistics
df_clean = df[df['CONTRACT WEIGHT'] <= 20]
df_outlier = df[df['CONTRACT WEIGHT'] > 20]

# Output outliers in file
file_path = 'data/Outlier.csv'
df_outlier.to_csv(file_path, index=False)

# Output clean dataset in file
file_path = 'data/Data_anonym_clean.csv'
df_clean.to_csv(file_path, index=False)
print(df_clean.shape)