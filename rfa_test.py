import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/sample.csv'
data = pd.read_csv(file_path)

# Convert dates to numerical format
reference_date = pd.Timestamp('2023-01-01')
for col in ['Created Date', 'Last Updated Date']:
    data[col] = pd.to_datetime(data[col])
    data[col] = (data[col] - reference_date).dt.days

# Define features and target variable
X = data.drop(['Priority', 'Test Case ID'], axis=1)  # Exclude non-predictive columns
y = data['Priority']


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing steps for the pipeline
# Assuming 'Description', 'Status', 'Assigned To', 'Created By' are correctly named as per your DataFrame
preprocessor = ColumnTransformer(
    transformers=[
        # Vectorize 'Description' and any other text fields
        ('text_vect_desc', TfidfVectorizer(), ['Description']),  # If multiple text fields, add them here
        # Encode categorical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Status', 'Assigned To', 'Created By']),
        # Scale date features
        ('date', StandardScaler(), ['Created Date', 'Last Updated Date'])
    ],
    remainder='passthrough'  # This will pass through other features without transformation
)


# Create the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
