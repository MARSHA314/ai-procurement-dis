import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

# Load dataframe
df_pre = pd.read_csv('data/Data_anonym_w_class.csv')

# Select only needed columns
columns_to_keep = [
    'QUARTER', 'YEAR','ID_VENDOR', 'ID_PLANT', 'MAT GROUP', 'VL_MPVS' ,'VL_BATCH',
    'ABC PRESENT', 'SAFETY TIME PROFILE', 'COVERAGE PROFILE SITE', 'COVERAGE PROFILE NO', 'SEGMENTATION',
   'LIFETIME', 'LAST WEEK SITE COMMENTS', 'SIP COMMENTS', 'SITE COMMENTS','BATCH', 'MPVS', 'FY','FY_LY', 'STD BATCH', 'FY WOW', 'VS CURRENT', 'VS PREV','AVG_COVERAGE_ONLY', 'AVG_COVERAGE_ONLY_Q2', 
   'TOTAL_DEMAND_ONLY', 'TOTAL_DEMAND_ONLY_Q2', 'AVG_DEMAND_ONLY', 'AVG_DEMAND_ONLY_Q2', 'TOTAL_PREBUILD', 'TOTAL_PREBUILD_Q2', 'AVG_PREBUILD', 
   'AVG_PREBUILD_Q2', 'AVG_INVENTORY_ONLY', 'AVG_INVENTORY_ONLY_Q2', 'AVG_SAFETY', 'AVG_SAFETY_Q2', 'AVG_SUPPLY', 'AVG_SUPPLY_Q2','CONTRACT CLASS' ]
df = df_pre.loc[:, columns_to_keep] 

# Output clean dataset in file
file_path = 'data/Data_training.csv'
df.to_csv(file_path, index=False)

# Set the feature matrix X and target vector y
X = df.drop(columns=['CONTRACT CLASS'])
y = df['CONTRACT CLASS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=15)

# List of categorical variables
col_categorical = [
 'ID_VENDOR', 'ID_PLANT', 'MAT GROUP', 'VL_MPVS' ,'VL_BATCH',
    'ABC PRESENT', 'SAFETY TIME PROFILE', 'COVERAGE PROFILE SITE', 'COVERAGE PROFILE NO', 'SEGMENTATION',
   'LIFETIME', 'LAST WEEK SITE COMMENTS', 'SIP COMMENTS', 'SITE COMMENTS',
]

# List of numerical variables
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

# Create the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=300, alpha=0.0001,
                    solver='adam', random_state=15, tol=1e-4)

# Train the model
mlp.fit(X_train, y_train)

# Predict on the test set with selected features
y_pred = mlp.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define the classes
classes = ['High', 'Low', 'Standard']

# Initialize the confusion matrix
confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

# Create a mapping from class names to indices
class_to_index = {cls: idx for idx, cls in enumerate(classes)}

# Populate the confusion matrix
for true_label, predicted_label in zip(y_test, y_pred):
    true_index = class_to_index[true_label]
    predicted_index = class_to_index[predicted_label]
    confusion_matrix[true_index, predicted_index] += 1

print(confusion_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

file_path = 'graphs/GRAPH_conf_matrix_plain.jpg'
plt.savefig(file_path, format='jpg', dpi=300)
plt.show()

# Plot training loss and accuracy
history = mlp.loss_curve_

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
file_path = 'graphs/GRAPH_Loss_plain.jpg'
plt.savefig(file_path, format='jpg', dpi=300)
plt.show()

# Save the model and scaler to files
joblib.dump(mlp, 'model/mlp_model_plain.pkl')
joblib.dump(preprocessor, 'model/preprocessor_plain.pkl')