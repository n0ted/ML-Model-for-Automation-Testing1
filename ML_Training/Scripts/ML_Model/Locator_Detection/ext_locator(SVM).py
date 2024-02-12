import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Assuming 'html_elements_locators.csv' is the output from the BeautifulSoup script
data = pd.read_csv('html_elements_locators.csv')

# Feature Engineering: Let's add a feature for text length
data['text'] = data['text'].fillna('').astype(str) 
data['TextLength'] = data['text'].apply(len)

X = data[['locator_type', 'locator_value', 'TextLength']]  # Adjust features here
y = data['tag_name']  # Predicting the tag name

# Function to transform locator_value into a numerical feature (could be improved)
def transform_locator_value(X):
    return X.apply(lambda x: len(x))

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['locator_type']),
        ('num', FunctionTransformer(transform_locator_value), ['locator_value']),
        ('txt_len', 'passthrough', ['TextLength'])
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
