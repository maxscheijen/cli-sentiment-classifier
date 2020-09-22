import pandas as pd
import glob

from pathlib import Path
from sklearn.model_selection import train_test_split


class Dataset:

    def load_data(self):
        data_paths = glob.glob("data/*.txt")

        self.data = pd.DataFrame()

        for data_file in data_paths:

            df = pd.read_csv(data_file,
                             delimiter='\t',
                             names=['sentence', 'sentiment'],
                             header=None)
            self.data = self.data.append(df)

        return self.data.reset_index(drop=True)

    def get_target(self, target_name):
        self.X = self.data.drop(target_name, axis=1).values.ravel()
        self.y = self.data[target_name]
        return self.X, self.y

    def split_data(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size,
                                                            stratify=self.y)
        return X_train, X_test, y_train, y_test
