import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(input_file, output_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file)

    # Perform preprocessing steps (e.g., handle missing values, convert categorical variables, etc.)
    # Example:
    # data.dropna(inplace=True)  # Drop rows with missing values
    # data['Category'] = pd.factorize(data['Category'])[0]  # Convert categorical variable to numerical

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['Target_Column'])
    y = data['Target_Column']

    # Define preprocessing transformations
    # Example: One-hot encode categorical variables, scale numerical variables, extract features from text
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Categorical_Column']),
            ('num', StandardScaler(), ['Numerical_Column']),
            ('text', TfidfVectorizer(), 'Text_Column')
        ],
        remainder='passthrough'
    )

    # Transform the features
    X_transformed = preprocessor.fit_transform(X)

    # Save the preprocessed features and target into a new CSV file
    preprocessed_data = pd.DataFrame(X_transformed)
    preprocessed_data['Target_Column'] = y
    preprocessed_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = 'c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/input.csv'  # Specify the input CSV file
    output_file = 'c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/preprocessed_data.csv'  # Specify the output preprocessed CSV file
    preprocess_data(input_file, output_file)
