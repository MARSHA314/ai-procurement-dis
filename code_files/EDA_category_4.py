import pandas as pd
import matplotlib.pyplot as plt
import os


# Load dataframe
df = pd.read_csv('data/Data_anonym_w_class.csv')
print(df.info())

# Select only CATEGORY columns with WEIGHT
columns_to_keep = [
   'YEAR', 'ID_PLANT', 'MAT GROUP',  'STD CONTRACT WEIGHT','LIFETIME', 'ABC PRESENT','SAFETY TIME PROFILE','COVERAGE PROFILE SITE', 'COVERAGE PROFILE NO',
   'SEGMENTATION','LAST WEEK SITE COMMENTS','SIP COMMENTS','SITE COMMENTS'
]
df_cat = df.loc[:, columns_to_keep] 

# Create subplots of boxplot for each category column
class_columns = ['YEAR', 'ID_PLANT', 'MAT GROUP', 'LIFETIME', 'ABC PRESENT','SAFETY TIME PROFILE','COVERAGE PROFILE SITE', 'COVERAGE PROFILE NO','SEGMENTATION',
   'LAST WEEK SITE COMMENTS','SIP COMMENTS','SITE COMMENTS']


# Loop through each class column and create/save the boxplot
for class_col in class_columns:
    fig, ax = plt.subplots(figsize=(6, 6))

    unique_classes = df_cat[class_col].unique()
    data_to_plot = [df_cat[df_cat[class_col] == cls]['STD CONTRACT WEIGHT'] for cls in unique_classes]

    # Create the boxplot
    ax.boxplot(data_to_plot, patch_artist=True)
    ax.set_title(f'Boxplot of VALUE by {class_col}')
    ax.set_xticklabels(unique_classes)
    ax.set_xlabel('Category')
    ax.set_ylabel('STD CONTRACT WEIGHT')

    save_folder = 'graphs/'
    save_path = os.path.join(save_folder,'boxplot_'+f'{class_col}.jpg')
    plt.savefig(save_path)

    # Close the figure to avoid memory issues
    plt.close(fig)