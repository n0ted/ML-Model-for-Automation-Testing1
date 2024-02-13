import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

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

# Load data
data = pd.read_csv('c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/sample.csv')
data = process_dates(data)

X = data.drop('Priority', axis=1)
y = data['Priority']

# We Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loading the trained model we can use pkl file also
model_pipeline = load_model(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        try:
            # Reading the CSV file
            df = pd.read_csv(file)
            
            # Process the new data 
            df = preprocess_new_test_cases(df)  
            
            # predictions using the loaded model
            predictions = model_pipeline.predict(df)
            
            # Return predictions as JSON using
            return jsonify({'predictions': predictions.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
