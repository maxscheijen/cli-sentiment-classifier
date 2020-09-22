import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, data_file):
        self.data_file = Path(data_file)

    def load_data(self):
        self.data = pd.read_csv(self.data_file,
                                delimiter='\t',
                                names=['sentence', 'sentiment'],
                                header=None)
        return self.data

    def get_target(self, target_name):
        self.X = self.data.drop(target_name, axis=1).values.ravel()
        self.y = self.data[target_name]
        return self.X, self.y

    def split_data(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size,
                                                            stratify=self.y)
        return X_train, X_test, y_train, y_test
