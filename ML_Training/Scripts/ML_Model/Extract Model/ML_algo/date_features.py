from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date=pd.Timestamp('2023-01-01')):
        self.reference_date = reference_date

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        # Ensure X is a DataFrame
        X = pd.DataFrame(X)
        # Convert date columns to datetime
        X['Created Date'] = pd.to_datetime(X['Created Date'])
        X['Last Updated Date'] = pd.to_datetime(X['Last Updated Date'])
        # Calculate the numerical value of dates and the difference
        X['Days Since Creation'] = (X['Created Date'] - self.reference_date).dt.days
        X['Days Since Last Update'] = (X['Last Updated Date'] - self.reference_date).dt.days
        X['Update Creation Difference'] = (X['Last Updated Date'] - X['Created Date']).dt.days
        # Drop original date columns
        return X.drop(['Created Date', 'Last Updated Date'], axis=1)