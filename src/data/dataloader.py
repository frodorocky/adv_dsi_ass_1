import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, path, test_size=0.2, random_state=42):
        self.path = path
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        self.df = pd.read_csv(self.path)
        # Extract the target variable into a variable called y
        self.y = self.df.pop('TARGET_5Yrs')
        # Set the index of a DataFrame called 'df' to the values of the column 'Id'.
        self.df = self.df.set_index(['Id'])
        return self.df, self.y

    def data_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y)
        X_train.to_csv("data/processed/X_train.csv", index=False)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)
        return X_train, X_test, y_train, y_test


