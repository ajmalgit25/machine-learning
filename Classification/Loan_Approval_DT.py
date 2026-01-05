# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the Dataset
dataset = pd.read_csv("../../Data_Files/Loan_Approval_Dataset.csv")

dataset.columns = dataset.columns.str.strip()   # Strip whitespaces from column names and values
for col in ['education', 'self_employed', 'loan_status']:
    dataset[col] = dataset[col].str.strip().str.lower()  # Convert to lowercase for consistency
#print(dataset.columns.tolist())

## Preprocessing the Data
# Drop loan_id (not useful for prediction)
dataset.drop("loan_id", axis=1, inplace=True)

# Handle Missing Values (if any)
dataset.dropna(inplace=True)  # Or use fillna if you want to impute


# ✅ Convert categorical columns to numeric using get_dummies
dataset = pd.get_dummies(dataset, columns=['education', 'self_employed'], drop_first=True)

# Extract Features and Target
x = dataset.drop("loan_status", axis=1)
y = dataset["loan_status"]


# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# Train the Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=4)     # Try 3, 4, or 5
model.fit(x_train, y_train)


# Make Predictions
y_pred = model.predict(x_test)


# Evaluate the Model
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion_matrix:\n", confusion_matrix(y_test, y_pred))


# Set figure size
plt.figure(figsize=(10, 5))

# Plot the tree
plot_tree(model, 
          feature_names=x.columns, 
          class_names=['Rejected', 'Approved'], 
          filled=True, 
          rounded=True, 
          fontsize=10)

# Show the plot
plt.title("Decision Tree - Loan Approval")
plt.show()
