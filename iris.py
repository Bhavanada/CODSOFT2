#!/usr/bin/env python
# coding: utf-8

# In[2]:


#IRIS PROJECT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris_data = pd.read_csv("C:\\Users\\compaq\\Desktop\\IRIS.csv")

# Display the first few rows of the dataset to understand its structure
print(iris_data.head())

# Define features and target
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (using K-Nearest Neighbors as an example)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)


# In[ ]:




