# Read data from the folders


import argparse
import logging
import random
from data_utils import DataLoader
from models import Model
import numpy as np
from adhoc_testing import adhoc_test

#######

# Parse input arguments from user

#######

parser = argparse.ArgumentParser(
    description='Sentiment analyzer on movie reviews.')
parser.add_argument('-f',
                    '--file', help='the directory that stores the test and train data. Each directory is assumed to have a subdirectory for postive and negative reviews.')
parser.add_argument(
    '-n', "--num_records", type=int, help="number of records", default=-1)

parser.add_argument(
    '-v', "--verbose", help="increase output verbosity", action="store_true")

parser.add_argument(
    '-i', "--interactive", help="increase output verbosity", action="store_true")


args = parser.parse_args()

logger = logging.getLogger('Sentiment Analysis')

logging_level = logging.DEBUG if args.verbose else logging.INFO

logging.basicConfig(level=logging_level)
# logging.debug('This message should go to the log file')
# logging.info('So should this')
# logging.warning('And this, too')

# format_string = "%(asctime)s %(filename)s:%(lineno)d %(funcName)s %(levelname)s %(name)s %(message)s"
# logging.basicConfig(level=logging_level, format=format_string)

if args.file is None:
  print('Need to pass the directory of reviews. Exiting...')
  parser.print_help()
  exit(-1)

dataLoader = DataLoader(logging)
training_data = dataLoader.load_data(
    args.file, 'train', limit=int(args.num_records), randomize=True)

logging.info('*************** describe data ***************')
logging.info(training_data.describe())

logging.info('*************** describe types *************** \n {0}'.format(
             training_data.dtypes))


logging.debug('head data set')
logging.debug(training_data.head(2))

logging.debug('tail data set')
logging.debug(training_data.tail(2))

# TODO(Abdul) - plot the bar graph of positive and negative reviews

# show a bar of total positive and negative reviews

# title = "Training Data"
# x-tick-labels = [negative, positive]
# xlabel = "kind of reviews"
# ylabel = "Count"

# Refer to this for code -
# https://matplotlib.org/gallery/statistics/barchart_demo.html

sentiments = np.array(training_data['sentiments'])
count_negative_reviews = (sentiments == False).sum()
count_positive_reviews = (sentiments == True).sum()
logging.info('training set:  neg reviews {} and pos reviews {}'.format(
    count_negative_reviews, count_positive_reviews))

if(args.interactive):
  input()

# TODO(Abdul) - plot the bar graph of avg length of review, avg length of
# positive and avg length of negative reviews
training_data['review_length'] = training_data['data'].str.len()

positive_reviews_df = training_data.loc[
    lambda df: df.sentiments == True]
logging.debug('positive review df')
logging.debug(positive_reviews_df.head())

negative_reviews_df = training_data.loc[
    lambda df: df.sentiments == False]
logging.debug('negative review df')
logging.debug(negative_reviews_df.head())

avg_length = training_data['review_length'].mean()
avg_positive_length = positive_reviews_df['review_length'].mean()
avg_negative_length = negative_reviews_df['review_length'].mean()
logging.info('training set: length avg {}, positive {} and negative {} '.format(
    avg_length, avg_positive_length, avg_negative_length))

print('******* Generating a Bayesian model*****************')
m = Model(name='bayes')
m.train(training_data.data, training_data.sentiments)


# Load test data
test_df = dataLoader.load_data(
    args.file, 'test', limit=int(args.num_records))
print('************ test data ******************')
print(test_df.head(2))
print(test_df.tail(2))

if(args.interactive):
  input()

# Generate predictions
predictions = m.predict(test_df.data)

# Analyse efficacy of the model
# TODO(Abdul) - make a line graph of predictions vs ground truth
m.analysis(test_df.sentiments, predictions)


print('******* Generating a SVM*****************')
if(args.interactive):
  input()

svm_model = Model(name='svm')
svm_model.train(training_data.data, training_data.sentiments)
predictions = svm_model.predict(test_df.data)
# analyze efficiency
svm_model.analysis(test_df.sentiments, predictions)


print('lets do some adhoc testing')
if(args.interactive):
  input()

# Testing with adhoc reviews from IMDB - for movie black panther
adhoc_test(m)
adhoc_test(svm_model)
