import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

data = pd.read_csv('annotated_dataset.csv')

data['HasChildren'] = data['HasChildren'].map({True: 1, False: 0})

X = data.drop('Element', axis=1)
y = data['Element']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['TextLength', 'HasChildren', 'Depth']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['TagName', 'Class'])
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', SVC(kernel='linear'))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
