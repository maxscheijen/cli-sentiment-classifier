mkdir -p data/

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip -O sentiment.zip

unzip sentiment.zip -d data/
mv data/sentiment\ labelled\ sentences/amazon_cells_labelled.txt data/amazon_reviews.txt
mv data/sentiment\ labelled\ sentences/imdb_labelled.txt data/imdb_reviews.txt
mv data/sentiment\ labelled\ sentences/yelp_labelled.txt data/yelp_reviews.txt

rm -rf data/__MACOSX sentiment.zip data/sentiment\ labelled\ sentences/