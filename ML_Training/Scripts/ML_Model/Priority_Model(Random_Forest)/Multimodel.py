import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import re

# Simulated code for data loading and preprocessing
# Assume 'data' is preprocessed DataFrame from the provided code snippet

data = pd.read_csv('c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/sample.csv')

def process_dates(df):
    reference_date = pd.Timestamp('2023-01-01')
    for col in ['Created Date', 'Last Updated Date']:
        df[col] = pd.to_datetime(df[col])
        df[col] = (df[col] - reference_date).dt.days
    return df

data = process_dates(data)

X = data.drop('Priority', axis=1)
y = data['Priority']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Status', 'Assigned To', 'Created By']  # Update based on your analysis
text_features = 'Description'  # Assuming we're only vectorizing the 'Description' for simplicity
date_features = ['Created Date', 'Last Updated Date']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('text', TfidfVectorizer(), text_features),
        ('date', 'passthrough', date_features)
    ],
    remainder='drop'
)
# Define ML models to evaluate
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=20, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(random_state=20, class_weight='balanced', max_iter=1000),
    'SVC': SVC(class_weight='balanced', probability=True, random_state=20),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=20, class_weight='balanced')
}



# Dictionary to store accuracy of each model
accuracy_scores = {}

# Assuming 'X_train', 'y_train', 'X_test', 'y_test' are defined based on the provided snippet
for name, model in models.items():
    # Update the model in the pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # 'preprocessor' defined in the provided code snippet
        ('classifier', model)
    ])
    
    # Train and predict
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy

# Plotting the accuracies
names = list(accuracy_scores.keys())
values = list(accuracy_scores.values())

plt.figure(figsize=(10, 5))
plt.bar(names, values)
plt.xlabel('Machine Learning Model')
plt.ylabel('Accuracy')
plt.title('ML Model Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()

# Note: This plotting code assumes matplotlib is available and the data processing steps are done correctly.
