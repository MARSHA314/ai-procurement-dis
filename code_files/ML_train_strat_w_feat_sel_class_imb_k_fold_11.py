import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, KFold

# Load dataframe
df = pd.read_csv('data/Data_training.csv')

# Set the feature matrix X and target vector y
X = df.drop(columns=['CONTRACT CLASS'])
y = df['CONTRACT CLASS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=15)

col_categorical = [
 'ID_VENDOR', 'ID_PLANT', 'MAT GROUP', 'VL_MPVS' ,'VL_BATCH',
    'ABC PRESENT', 'SAFETY TIME PROFILE', 'COVERAGE PROFILE SITE', 'COVERAGE PROFILE NO', 'SEGMENTATION',
   'LIFETIME', 'LAST WEEK SITE COMMENTS', 'SIP COMMENTS', 'SITE COMMENTS',
]

col_numerical = [
   'BATCH', 'MPVS', 'FY','FY_LY', 'STD BATCH', 'FY WOW', 'VS CURRENT', 'VS PREV','AVG_COVERAGE_ONLY', 'AVG_COVERAGE_ONLY_Q2', 
   'TOTAL_DEMAND_ONLY', 'TOTAL_DEMAND_ONLY_Q2', 'AVG_DEMAND_ONLY', 'AVG_DEMAND_ONLY_Q2', 'TOTAL_PREBUILD', 'TOTAL_PREBUILD_Q2', 'AVG_PREBUILD', 
   'AVG_PREBUILD_Q2', 'AVG_INVENTORY_ONLY', 'AVG_INVENTORY_ONLY_Q2', 'AVG_SAFETY', 'AVG_SAFETY_Q2', 'AVG_SUPPLY', 'AVG_SUPPLY_Q2'
]

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), col_numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), col_categorical)
    ])

# Fit and transform the training data, transform the test data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Apply SMOTE to the training data with selected features
smote = SMOTE(random_state=15)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Create the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=500, alpha=0.0001,
                    solver='adam', random_state=15, tol=1e-4)

# Train the model
mlp.fit(X_train, y_train)

# Get the feature importances (using the absolute value of the weights)
importances = np.abs(mlp.coefs_[0]).sum(axis=1)

# Select features based on importance
threshold = np.percentile(importances, 25)  
selected_features = np.where(importances >= threshold)[0]

# Transform the dataset to keep only selected features
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Re-create the MLPClassifier to reset it
mlp_selected = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=300, alpha=0.0001,
                    solver='adam', random_state=15, tol=1e-4)

# Define k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=15)

# Perform cross-validation
cv_scores = cross_val_score(mlp_selected, X, y, cv=kf, scoring='accuracy')

# Print the results
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")
print(f"Standard deviation of cross-validation score: {cv_scores.std()}")


