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

data = pd.read_csv('c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/sample.csv')

# Function to process dates
def process_dates(df):
    reference_date = pd.Timestamp('2023-01-01')
    for col in ['Created Date', 'Last Updated Date']:
        df[col] = pd.to_datetime(df[col])
        df[col] = (df[col] - reference_date).dt.days
    return df

# Preprocess data
data = process_dates(data)

# Define mapping of categories to importance levels
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

# Replace category values in 'Module' column with importance levels
data['Module Importance'] = data['Module'].map(module_importance)
data.dropna(inplace=True)


# Get the indices of dropped rows


# Split features and target
X = data.drop(['Priority', 'Module'], axis=1)  # Remove 'Module' since we have 'Module Importance'
y = data['Priority']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


categorical_features = ['Status', 'Assigned To', 'Created By', 'Bug', 'Change Frequency', 'Environment']
text_features = 'Description'
date_features = ['Created Date', 'Last Updated Date']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(), text_features),
        ('date', 'passthrough', date_features),
        ('mod_imp', 'passthrough', ['Module Importance'])  # Include 'Module Importance' as a passthrough feature
    ],
    remainder='drop'
)
#  ML models 
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=20, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(random_state=20, class_weight='balanced', max_iter=10000),
    'SVC': SVC(class_weight='balanced', probability=True, random_state=20),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=20, class_weight='balanced')
}



#storing accuracy values of all the algorithms
accuracy_scores = {}


for name, model in models.items():
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('classifier', model)
    ])
    
    # Train and predict
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy

# Plotting on graph
names = list(accuracy_scores.keys())
values = list(accuracy_scores.values())

plt.figure(figsize=(10, 5))
plt.bar(names, values)
plt.xlabel('Machine Learning Model')
plt.ylabel('Accuracy')
plt.title('ML Model Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()

# Graphical Representation 
plt.show()  

# Note: This plotting code assumes matplotlib is available and the data processing steps are done correctly.
