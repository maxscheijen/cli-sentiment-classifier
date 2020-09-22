"""MLP - machine-learning-production
Usage:
    mlp-cli train <dataset-file>
    mlp-cli ask <question>
    mlp-cli (-h | --help)
Arguments:
    <dataset-dir>  Directory with dataset.
Options:
    -h --help                  Show this screen.
"""

from docopt import docopt
from mlp.dataset import Dataset
from mlp.model import Model


def train_model(data_file):
    print(f"Training model from {data_file}")
    data = Dataset(data_file)
    data.load_data()
    data.get_target("sentiment")
    X_train, X_test, y_train, y_test = data.split_data(0.2)

    model = Model()
    model.train(X_train, y_train)
    model.save_model()


def ask_model(question):
    model = Model().load_model()

    print(f"Predicted sentiment about the following statement: {question}")
    y_pred = model.predict_proba([question])
    print(f"Negative: {y_pred[0][0]:.2f} \nPositive: {y_pred[0][1]:.2f}")


def main():
    arguments = docopt(__doc__)

    if arguments["train"]:
        train_model(arguments['<dataset-file>'])

    elif arguments["ask"]:
        ask_model(arguments["<question>"])


if __name__ == "__main__":
    main()
