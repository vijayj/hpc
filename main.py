# Read data from the folders


import argparse
import logging
import random
from data_utils import DataLoader
import numpy as np

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


args = parser.parse_args()

logger = logging.getLogger('Sentiment Analysis')

logging_level = logging.DEBUG if args.verbose else logging.INFO

logging.basicConfig(level=logging_level)
# logging.debug('This message should go to the log file')
# logging.info('So should this')
# logging.warning('And this, too')

#format_string = "%(asctime)s %(filename)s:%(lineno)d %(funcName)s %(levelname)s %(name)s %(message)s"
#logging.basicConfig(level=logging_level, format=format_string)

if args.file is None:
  print('Need to pass the directory of reviews. Exiting...')
  parser.print_help()
  exit(-1)


training_data = DataLoader(logging).load_data(
    args.file, 'train', limit=int(args.num_records), randomize=True)

logging.info('describe data')
logging.info(training_data.describe())

logging.info('describe types')
logging.info(training_data.dtypes)

logging.debug('head data set')
logging.debug(training_data.head(2))

logging.debug('tail data set')
logging.debug(training_data.tail(2))

# TODO(Abdul) - plot the bar graph of positive and negative reviews

# show a bar of total positive and negative reviews

#title = "Training Data"
# x-tick-labels = [negative, positive]
# xlabel = "kind of reviews"
# ylabel = "Count"

# Refer to this for code -
# https://matplotlib.org/gallery/statistics/barchart_demo.html

sentiments = np.array(training_data['sentiments'])
count_negative_reviews = (sentiments == False).sum()
count_positive_reviews = (sentiments == True).sum()
logging.debug('counts neg {} and pos {}'.format(
    count_negative_reviews, count_positive_reviews))

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
logging.debug('length avg {}, positive {} and negative {} '.format(
    avg_length, avg_positive_length, avg_negative_length))


# Run the bayesian classifier on training data
