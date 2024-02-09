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

# from bs4 import BeautifulSoup
# import pandas as pd

# # Assuming html_content contains your HTML
# with open('ML_Training/DOM Files/Test1/site1.html', 'r', encoding='utf-8') as file:
#     html_content = file.read()

# soup = BeautifulSoup(html_content, 'html.parser')

# data = []

# # Function to extract features for any given element
# def extract_features(element, element_name):
#     return {
#         'Element': element_name,
#         'TagName': element.name,
#         'Class': ' '.join(element.get('class', '')),
#         'TextLength': len(element.text.strip()),
#         'HasChildren': len(element.find_all(recursive=False)) > 0,
#         'Depth': len(element.find_parents())
#     }

# # List of element types and their classes to iterate over
# elements_to_find = [
#     {'class': 'header', 'name': 'Header'},
#     {'class': 'navbar', 'name': 'Navbar Link'},
#     {'class': 'main', 'name': 'Main Content Heading'},
#     {'class': 'footer', 'name': 'Footer'}
# ]

# for element in elements_to_find:
#     for found_element in soup.find_all(class_=element['class']):
#         data.append(extract_features(found_element, element['name']))

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save to CSV
# df.to_csv('annotated_dataset.csv', index=False)
