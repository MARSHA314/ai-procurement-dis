import pandas as pd
import numpy as np
import pickle

# Load the dataset
df = pd.read_csv('data/Data_sample.csv')

print(df.info())
print(df.head)

# Function to anonymize a column
def anonymize_column(column):
    unique_values = column.unique()
    mapping = {value: idx for idx, value in enumerate(unique_values, start=1)}
    anonymized_column = column.map(mapping)
    return anonymized_column, mapping

#Remove data where SIP information is missing
df = df[df['SEGMENTATION'].notna()]

# Fill 0 where nan
df['FY_LY'] = df['FY_LY'].fillna(df['FY'])
df['FY WOW'] = df['FY WOW'].fillna(0)
df['VS CURRENT'] = df['VS CURRENT'].fillna(0)
df['VS PREV'] = df['VS PREV'].fillna(0)

# List of SIP columns to fill NaN with 0
columns_to_fill = ['AVG_COVERAGE_ONLY', 'AVG_COVERAGE_ONLY_Q2', 'TOTAL_DEMAND_ONLY', 'TOTAL_DEMAND_ONLY_Q2', 
                   'AVG_DEMAND_ONLY', 'AVG_DEMAND_ONLY_Q2', 'TOTAL_PREBUILD', 'TOTAL_PREBUILD_Q2', 'AVG_PREBUILD', 
                   'AVG_PREBUILD_Q2', 'AVG_INVENTORY_ONLY', 'AVG_INVENTORY_ONLY_Q2', 'AVG_SAFETY', 'AVG_SAFETY_Q2', 'AVG_SUPPLY', 'AVG_SUPPLY_Q2']

# Replace NaN with 0 in the specified columns
df[columns_to_fill] = df[columns_to_fill].fillna(0)

# Create Quarter column
conditions = [
    df['WEEK'] <= 13,
    (df['WEEK'] > 13) & (df['WEEK'] <= 26),
    (df['WEEK'] > 26) & (df['WEEK'] <= 39),
    df['WEEK'] > 39
]
choices = [1,2,3,4]
df['QUARTER'] = np.select(conditions, choices, default=0)


# Remove obsolete columns
columns_to_keep = [
   'YEAR', 'WEEK','QUARTER', 'ID_MATERIAL', 'ID_VENDOR', 'ID_PLANT', 'MAT GROUP', 'CONTRACT QTY', 'PRICE', 
   'CD_DOWN_INTERVAL', 'VL_BATCH', 'VL_MPVS', 'VL_MIN_ORDER_QTY', 'BATCH', 'MPVS', 'FY', 'FY_LY',
   'STD BATCH', 'FY WOW','SUPER THEME', 'ABC PRESENT', 'SAFETY TIME PROFILE', 
   'COVERAGE PROFILE SITE','COVERAGE PROFILE NO', 'VS CURRENT', 'VS PREV', 'SEGMENTATION',  'LIFETIME', 
   'LAST WEEK SITE COMMENTS', 'SIP COMMENTS', 'SITE COMMENTS', 'AVG_COVERAGE_ONLY', 
   'AVG_COVERAGE_ONLY_Q2', 'TOTAL_DEMAND_ONLY', 'TOTAL_DEMAND_ONLY_Q2', 'AVG_DEMAND_ONLY', 
   'AVG_DEMAND_ONLY_Q2', 'TOTAL_PREBUILD', 'TOTAL_PREBUILD_Q2', 'AVG_PREBUILD', 'AVG_PREBUILD_Q2', 
   'AVG_INVENTORY_ONLY', 'AVG_INVENTORY_ONLY_Q2', 'AVG_SAFETY', 'AVG_SAFETY_Q2', 'AVG_SUPPLY', 'AVG_SUPPLY_Q2'
]
df = df.loc[:, columns_to_keep] 
print(df.shape)

# Anonymize relevant columns
columns_to_anonymize = ['ID_MATERIAL', 'ID_VENDOR','ID_PLANT','SAFETY TIME PROFILE','MAT GROUP', 'SEGMENTATION', 'ABC PRESENT', 'SUPER THEME', 'COVERAGE PROFILE SITE',
                        'COVERAGE PROFILE NO','LAST WEEK SITE COMMENTS','SIP COMMENTS','SITE COMMENTS']

# Dictionary to store mappings
mappings = {}

# Anonymize relevant columns
for column in columns_to_anonymize:
    df[column], mapping = anonymize_column(df[column])
    mappings[column] = mapping


# Save the anonymized dataset to a new CSV file
csv_file_path = 'data/Data_anonym.csv'
df.to_csv(csv_file_path, index=False)


# Save mappings to a file
with open('model/mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)

