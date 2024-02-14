import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
data = pd.read_csv('c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/sample.csv')

text_columns = ['Description', 'Steps to Reproduce', 'Preconditions', 'Expected Results', 'Actual Results']
categorical_columns = ['Status', 'Assigned To', 'Created By']

# Tokenization for text columns
for column in text_columns:
    count_vectorizer = CountVectorizer()
    tokenized_data = count_vectorizer.fit_transform(data[column])
    tokenized_df = pd.DataFrame(tokenized_data.toarray(), columns=count_vectorizer.get_feature_names_out())
    data = pd.concat([data, tokenized_df], axis=1)

# One-hot encoding for categorical columns
one_hot_encoder = OneHotEncoder(drop='first')  # drop='first' to avoid dummy variable trap
encoded_categorical_data = one_hot_encoder.fit_transform(data[categorical_columns])
encoded_categorical_df = pd.DataFrame(encoded_categorical_data.toarray(), columns=one_hot_encoder.get_feature_names_out())
data = pd.concat([data, encoded_categorical_df], axis=1)


# Drop original text and categorical columns
data.drop(columns=text_columns + categorical_columns, inplace=True)

# Saving preprocessed data
data.to_csv('c:/Users/Lenovo/Downloads/Test/ML_Training/DataSet/preprocessed_dataset.csv', index=False)
