import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

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

# Apply SMOTE to the data 
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

# Save list of features into a file together with weights
all_feature_names = preprocessor.get_feature_names_out()
selected_feature_names = [all_feature_names[i] for i in selected_features]
selected_importances = importances[selected_features]
selected_features_df = pd.DataFrame({
    'Feature Index': selected_features,
    'Feature Name': selected_feature_names,
    'Importance': selected_importances
})
selected_features_df.to_csv('graphs/selected_features_importances_w_SMOTE.csv', index=False)

# Re-create the MLPClassifier to reset it
mlp_selected = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=300, alpha=0.0001,
                    solver='adam', random_state=15, tol=1e-4)

# Train the model again with selected features
mlp_selected.fit(X_train_selected, y_train)

# Predict on the test set with selected features
y_pred = mlp_selected.predict(X_test_selected)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Draw confusion matrix
classes = ['High', 'Low', 'Standard']
confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
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

# Save the plot as a JPEG file to the specified folder
file_path = 'graphs/GRAPH_conf_matrix_feat_sel_smote.jpg'
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
file_path = 'graphs/GRAPH_Loss_feat_sel_smote.jpg'
plt.savefig(file_path, format='jpg', dpi=300)
plt.show()

# Save the model and scaler to files
joblib.dump(mlp_selected, 'model/mlp_model_feat_sel_smote.pkl')
joblib.dump(preprocessor, 'model/preprocessor_feat_sel_smote.pkl')


