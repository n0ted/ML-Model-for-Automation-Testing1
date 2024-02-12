import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import re


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

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(random_state=20, class_weight='balanced'))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
try:
    with open('acc_report.txt', 'r') as f:
        content = f.read()
        # Use a regular expression to find all occurrences of "Trial <number>:" and extract <number>
        trial_numbers = re.findall(r'Trial (\d+):', content)
        if trial_numbers:
            # Convert found numbers to integers and take the max to find the last trial number
            last_trial_num = max(map(int, trial_numbers))
        else:
            last_trial_num = 0
except FileNotFoundError:
    last_trial_num = 0

trial_num = last_trial_num + 1

file_path = 'C:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/acc_report.txt'
with open(file_path, 'a') as f:
    print(f"Trial {trial_num}:", file=f)
    print("Accuracy report based on adjusted weights\n", file=f)
    print("Accuracy:", accuracy, file=f)
    print("Classification Report:\n", classification_report(y_test, y_pred), file=f)
    print("\n---\n", file=f)