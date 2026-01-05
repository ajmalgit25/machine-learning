import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)
 
confusion = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n {report}")
print(f"Confusion Matrix:\n {confusion}")


# Plot the data and logistic regression curve
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, model.predict_proba(X)[:, 1], color='red', label='Logistic Curve')
plt.xlabel('X - Features')
plt.ylabel('Probability')
plt.title('Logistic Regression')
plt.legend()
plt.show()

