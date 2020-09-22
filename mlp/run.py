"""MLP - machine-learning-production
Usage:
    mlp-cli train [--vocab-size=<vocab-size>]
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


def train_model(vocab_size):
    print("Training model")
    data = Dataset()
    data.load_data()
    data.get_target("sentiment")
    X_train, X_test, y_train, y_test = data.split_data(0.2)

    model = Model(vocab_size)
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
        train_model(int(arguments['--vocab-size']))

    elif arguments["ask"]:
        ask_model(arguments["<question>"])


if __name__ == "__main__":
    main()
