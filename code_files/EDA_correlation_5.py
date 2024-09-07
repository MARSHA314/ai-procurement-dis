import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataframe
df = pd.read_csv('data/Data_anonym_w_class.csv')
print(df.info())

# Select only TIME columns relevant to contract weight
columns_to_keep = [
   'STD CONTRACT WEIGHT','YEAR', 'WEEK', 'ID_PLANT' , 'ID_VENDOR', 'MAT GROUP',
]
df_time = df.loc[:, columns_to_keep] 

# Correlation matrix
corr = df_time.corr()

# Save the matrix to csv file
csv_file_path = 'graphs/Correlation_time.csv'
corr.to_csv(csv_file_path, index=False)

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",annot_kws={"size": 8})
plt.title('Correlation Matrix Heatmap')

file_path = 'graphs/Correlation_heatmap_time_CW.jpg'
plt.savefig(file_path, format='jpg', dpi=300)
plt.show()

# Select only CONTRACT columns
columns_to_keep = [
    'STD CONTRACT WEIGHT', 'CONTRACT QTY', 'PRICE', 'CD_DOWN_INTERVAL', 'BATCH', 'MPVS', 'FY', 'FY_LY', 'STD BATCH', 'FY WOW'
]
df_contract = df.loc[:, columns_to_keep] 

corr = df_contract.corr()

csv_file_path = 'graphs/Correlation_contract.csv'
corr.to_csv(csv_file_path, index=False)

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",annot_kws={"size": 8})
plt.title('Correlation Matrix Heatmap')

file_path = 'graphs/Correlation_heatmap_contract_CW.jpg'
plt.savefig(file_path, format='jpg', dpi=300)

plt.show()

# Select only SIP columns
columns_to_keep = [
    'STD CONTRACT WEIGHT', 'VS CURRENT', 'VS PREV', 
    'AVG_COVERAGE_ONLY', 'AVG_COVERAGE_ONLY_Q2', 'TOTAL_DEMAND_ONLY', 
    'TOTAL_DEMAND_ONLY_Q2','AVG_INVENTORY_ONLY', 'AVG_INVENTORY_ONLY_Q2',
    'AVG_SUPPLY', 'AVG_SUPPLY_Q2'
]
df_sip = df.loc[:, columns_to_keep] 

# Correlation matrix
corr = df_sip.corr()

csv_file_path = 'graphs/Correlation_sip.csv'
corr.to_csv(csv_file_path, index=False)

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",annot_kws={"size": 8})
plt.title('Correlation Matrix Heatmap')

file_path = 'graphs/Correlation_heatmap_sip_CW.jpg'
plt.savefig(file_path, format='jpg', dpi=300)

plt.show()