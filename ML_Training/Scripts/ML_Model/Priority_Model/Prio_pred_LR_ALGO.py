import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
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

categorical_features = ['Status', 'Assigned To', 'Created By']
text_features = 'Description'
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
    ('logistic_regression', LogisticRegression(max_iter=1000))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

def process_dates(df):
    reference_date = pd.Timestamp('2023-01-01')
    for col in ['Created Date', 'Last Updated Date']:
        df[col] = pd.to_datetime(df[col])
        df[col] = (df[col] - reference_date).dt.days
    return df

def get_preprocessor():
    categorical_features = ['Status', 'Assigned To', 'Created By']
    text_features = 'Description'
    date_features = ['Created Date', 'Last Updated Date']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('text', TfidfVectorizer(), text_features),
            ('date', 'passthrough', date_features)
        ],
        remainder='drop'
    )
    return preprocessor

# Load logistic regression model
def load_model():
    model_pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('logistic_regression', LogisticRegression(max_iter=1000))
    ])
    # Assuming your model is saved in a file named 'logistic_regression_model.pkl'
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

# Function to preprocess new test cases
def preprocess_new_test_cases(new_test_cases):
    # Preprocess new test cases similar to how you preprocess the training data
    new_test_cases = process_dates(new_test_cases)
    return new_test_cases

# Function to predict priority for new test cases
def predict_priority_for_new_test_cases(new_test_cases):
    # Preprocess new test cases
    new_test_cases = preprocess_new_test_cases(new_test_cases)
    # Predict priority
    new_test_case_predictions = model_pipeline.predict(new_test_cases)
    return new_test_case_predictions

# Load data
data = pd.read_csv('c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/sample.csv')
data = process_dates(data)
X = data.drop('Priority', axis=1)
y = data['Priority']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model_pipeline = load_model()

while True:
    # User input for new test case
    new_test_case = pd.DataFrame(columns=X_train.columns)
    for col in X_train.columns:
        value = input(f"Enter value for {col}: ")
        new_test_case[col] = [value]
    
    # Predict priority for new test case
    predicted_priority = predict_priority_for_new_test_cases(new_test_case)
    
    # Output prediction
    print(f"Predicted Priority: {predicted_priority[0]}")
    another_test_case = input("Do you want to input another test case? (yes/no): ")
    if another_test_case.lower() != 'yes':
        break
