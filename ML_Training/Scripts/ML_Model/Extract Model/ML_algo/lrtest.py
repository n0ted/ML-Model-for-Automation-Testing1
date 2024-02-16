import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from preprocess import process_dates

module_importance = {
    'Authentication': 5,
    'User Profile': 4,
    'Registration': 3,
    'Password Reset': 3,
    'Access Control': 4,
    'Security': 5,
    'Blog': 2,
    'Search': 3,
    'Accessibility': 4,
    'Audit Trail': 3,
    'Settings Management': 3,
    'File Management': 2,
    'Localization': 4,
    'Data Management': 3,
    'Quality Assurance': 5,
    'User Experience': 4,
    'Marketing': 2,
    'Performance': 5,
    'Notification Management': 3,
    'Content Management': 3,
    'Service Availability': 4,
    'Time Management': 3,
    'Content Consumption': 2,
    'Integration Testing': 4,
    'Performance Testing': 5,
    'Accessibility Testing': 4,
    'Security Testing': 5,
    'Disaster Recovery': 4,
    'Authentication Testing': 5,
    'API Testing': 5,
    'Localization Testing': 4,
    'User Profile Management': 4,
    'Social Media Integration': 3,
    'Email Management': 3,
    'UI/UX Testing': 4,
    'Session Management': 3,
    'Compatibility Testing': 3,
    'Notification System Testing': 3,
    'Disaster Recovery Testing': 4,
    'Validation Testing': 3,
    'Feedback System Testing': 3,
    'Network Connectivity': 3,
    'Notification System': 2,
    'Date/Time Handling': 2,
    'User Account Management': 4
}

# Load the trained model
model_pipeline = joblib.load('c:/Users/Lenovo/Downloads/Test/ML_Training/Scripts/ML_Model/Extract Model/Model/trained_model1.joblib')

# Define function to preprocess dates
def process_dates(df):
    reference_date = pd.Timestamp('2023-01-01')
    for col in ['Created Date', 'Last Updated Date']:
        df[col] = pd.to_datetime(df[col])
        df[col] = (df[col] - reference_date).dt.days
    return df

# Define function to preprocess user input
def preprocess_input(user_input):
    user_df = pd.DataFrame([user_input])
    user_df['Module Importance'] = user_df['Module'].map(module_importance)
    user_df.drop(['Module', 'Test Case ID', 'Assigned To', 'Created By'], axis=1, inplace=True)
    return user_df

# Function to predict priority
def predict_priority(user_input):
    user_df = preprocess_input(user_input)
    priority = model_pipeline.predict(user_df)[0]
    return priority

# Example user input
user_input = {
    'Test Case ID': 'TC148',
    'Description': "Validate push notification settings functionality",
    'Assigned To': 'QA1',
    'Created By': 'QA3',
    'Status': 'Passed',
    'Steps to Reproduce': 'Adjust push notification settings in the application',
    'Preconditions': 'Push notification settings are customizable',
    'Expected Results': 'User is able to customize push notification settings successfully',
    'Actual Results': 'To Be Tested',
    'Created Date': '2023-05-28',
    'Last Updated Date': '2023-05-28',
    'Module': 'Notification System',
    'Bug': 'No',
    'Change Frequency': 'Daily',
    'Environment': 'Staging'
}



# Predict priority
predicted_priority = predict_priority(user_input)
print("Predicted Priority:", predicted_priority)
