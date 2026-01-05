import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
# Sample data
#X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
#y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

dataset = pd.read_csv("../../Data_Files/Breast_Cancer.csv")

#print (dataset.columns)

# Preprocessing the Data (Handle missing values (if any))
dataset.replace('?', np.nan, inplace=True) # Convert '?' to NaN (common in medical datasets)
dataset.dropna(inplace=True)  # Remove rows with missing values


# Convert 'Bare Nuclei' column to numeric (since it's likely in string format)
dataset['Bare Nuclei'] = pd.to_numeric(dataset['Bare Nuclei'])


# Split features (X) and target (y)
X = dataset.drop(columns=['Class'])  # Features
y = dataset['Class']  # Target (0 = benign, 1 = malignant)


# If Class values are not binary (e.g., 2 & 4 instead of 0 & 1), normalize them
y = y.map({2: 0, 4: 1})  # Mapping 2 to 0 (benign), 4 to 1 (malignant)


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalize the feature values (recommended for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)


# Make Predictions
y_pred = model.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)

# Step 9: Visualize Confusion Matrix (optional)
import seaborn as sns


plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Predict probabilities instead of class labels
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of being in class 1 (cancer)

# Add cancer probability to output
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Cancer Probability': y_prob})

# Display results
print("\nPredictions with Cancer Probability:")
print(results.head(10))  # Show first 10 predictions

##


