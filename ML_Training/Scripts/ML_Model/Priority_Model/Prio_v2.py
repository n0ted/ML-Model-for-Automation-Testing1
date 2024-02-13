import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def process_dates(df):
    reference_date = pd.Timestamp('2023-01-01')
    for col in ['Created Date', 'Last Updated Date']:
        df[col] = (pd.to_datetime(df[col]) - reference_date).dt.days
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


def load_model(X_train, y_train):
    model_pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('logistic_regression', LogisticRegression(max_iter=1000))
    ])
    model_pipeline.fit(X_train, y_train)
    return model_pipeline


def preprocess_new_test_cases(new_test_cases):
    new_test_cases = process_dates(new_test_cases)
    return new_test_cases


def predict_priority_for_new_test_cases(new_test_cases):
    new_test_cases = preprocess_new_test_cases(new_test_cases)
    return model_pipeline.predict(new_test_cases)


# Load data
data = pd.read_csv('c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/sample.csv')
data = process_dates(data)

X = data.drop('Priority', axis=1)
y = data['Priority']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model_pipeline = load_model(X_train, y_train)

# Predict priority for new test cases
new_test_cases = pd.read_csv('ML_Training/DataSet/input.csv')
predicted_priorities = predict_priority_for_new_test_cases(new_test_cases)

# Output predictions

new_test_cases['Predicted Priority'] = predicted_priorities
print(new_test_cases)

file_path = 'ML_Training/DataSet/genoutput.csv'
with open(file_path, 'a') as f:
    print(new_test_cases, file = f)
