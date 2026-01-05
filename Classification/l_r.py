import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv("../../Data_Files/Loan_Approval_Dataset.csv")
#print(dataset)
dataset.columns = dataset.columns.str.strip()
for col in ['education', 'self_employed', 'loan_status']:
    dataset[col] = dataset[col].str.strip().str.lower()


dataset.drop("loan_id", axis=1, inplace=True)
dataset.dropna(inplace=True)

dataset = pd.get_dummies(dataset, columns=['education', 'self_employed'], drop_first=True)

X = dataset.drop("loan_status", axis=1)
y = dataset["loan_status"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_pred, y_test))
print("Classification Report:", classification_report(y_pred, y_test))
