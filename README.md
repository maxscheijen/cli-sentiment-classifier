# Sentiment Classifier

This repo contains the code for training a sentiment classifier in the command line. You can also use the command line to predict the sentiment of your own sentences.

## Install instructions

These are the instructions to download the data, train the sentiment classifier, and predict on your own sentences.

Type these commands into your terminal. Make sure to install python 3.6 or greater.

### Step 1 - Get code and download data

``` shell
# Clone this repo
git clone https://github.com/maxscheijen/cli-sentiment-classifier.git

# Into directory
cd cli-sentiment-classifier

# Allow executing of shell script
chmod +x ./download_data.sh

# Download data
./download_data.sh
```

### Step 2 - Setup virtual python environment

```shell
# Create environment
python3 -m venv venv
source venv/bin/activate
```

### Step 3 - Install python packages

```shell
# Install required packages
pip install -r requirements.txt
```

### Step 4 - Install local package

Install local as specified in `setup.py` using the following command:

```shell
pip install -e .
```

This command allows us the `mlp-cli` for interactive command line interface.

### Step 5 - Train model on dataset

The command below trains the sentiment classifier on the Amazon dataset. The trained pickle files are stored in the `models` directory.

```shell
mlp-cli train data/amazon_reviews.txt
```

### Step 6 - Predict sentiment on new sentences

```shell
mlp-cli ask "This is good product."
```
