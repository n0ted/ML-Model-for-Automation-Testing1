import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import torch
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Read data
data = pd.read_csv('c:/Users/Lenovo/Downloads/Test/ml-predictions-app/preprocessed_dataset.csv')

# Function to process dates
def process_dates(df):
    reference_date = pd.Timestamp('2023-01-01')
    for col in ['Created Date', 'Last Updated Date']:
        df[col] = pd.to_datetime(df[col])
        df[col] = (df[col] - reference_date).dt.days
    return df

# Preprocess data
data = process_dates(data)

# Split features and target
X = data.drop('Priority', axis=1)
y = data['Priority']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical, text, and date features
categorical_features = ['Status', 'Assigned To', 'Created By']
text_features = 'Description'
date_features = ['Created Date', 'Last Updated Date']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('text', TfidfVectorizer(), text_features),
        ('date', 'passthrough', date_features)
    ],
    remainder='drop'
)

# Logistic Regression model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logistic_regression', LogisticRegression(max_iter=1000))
])

# Train model
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:/n", classification_report(y_test, y_pred))
print(X_train.dtypes)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader objects
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model architecture
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define model parameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1

# Initialize model, loss function, and optimizer
model = MyModel(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))  # Assuming labels are 1D, add unsqueeze to match output shape
        loss.backward()
        optimizer.step()

# Evaluating
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    predictions = (torch.sigmoid(outputs) > 0.5).float()  # Example thresholding for binary classification
    accuracy = (predictions == y_test_tensor).float().mean().item()


print("Accuracy:", accuracy)