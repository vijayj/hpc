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
logging.info('training set: counts neg {} and pos {}'.format(
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
logging.info('training set: length avg {}, positive {} and negative {} '.format(
    avg_length, avg_positive_length, avg_negative_length))


# TODO(VJ) - cleanup - move models into different files

# Run the bayesian classifier on training data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# create a classifier
bayesian_review_classifier = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# Train the bayesian_model
bayesian_model = bayesian_review_classifier.fit(
    training_data.data, training_data.sentiments)

# Test accuracy
test_df = dataLoader.load_data(
    args.file, 'test', limit=int(args.num_records))
print(test_df.head())
print(test_df.tail())

predictions = bayesian_model.predict(test_df.data)
print("Accurancy (bayes model): ", np.mean(predictions ==
                                           test_df.sentiments))

# Let's now try and do ratings as category instead of just booleans
# We will use a SVM and encode the ratings category so that we can use SVM
# The results of predictions will have to be decoded to the ratings...
# Train a SVM
from sklearn.linear_model import SGDClassifier
svm_model = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1e-3,
        random_state=42,
        max_iter=5,
        tol=None))
])
svm_model.fit(training_data.data, training_data.sentiments)
predicted = svm_model.predict(test_df.data)
np.mean(predicted == test_df.sentiments)
print("Accurancy (svm model): ", np.mean(predicted == test_df.sentiments))

from sklearn import metrics
print('***svm report***')
print(metrics.classification_report(test_df.sentiments,
                                    predicted,
                                    target_names=['True', 'False']))

# print('***confusion matrix***')
# print(metrics.confusion_matrix(test_df.sentiments, predicted))


# logging.info('doing cross validation with grid search to optimize svm model')
####
# Testing with adhoc reviews from IMDB - for movie black panther
# store the model and try with some other reviews

r1 = '''
When I went to see this movie, I had a general positive expectation. Maybe Marvel movies are not at the top of my wish list, but I was curious enough to give it a willing try. Besides, it was a holiday and I had time to spend with my kid.

Boy, was I disappointed.

I found the story to be far-fetched, schematic, and uninvolving. Maybe that last quality is its greatest sin. I can deal with traditional story motives, good vs. bad, your occasional plot twist (to be seen miles ahead), but there must be more to a movie to grasp an audience's emotions. This movie did not do that for me, and seeing the lukewarm attention that it got for the other people in the cinema, it did not do the trick for them either.

What didn't particularly help was the wooden acting with the stiffest wood grain. It is as if most actors read their lines from a visual script, and were too busy doing that to offer more than high-school level acting. The only sad excuse may be that this movie is a derivative of a comic, so the dialogue was exactly as stiff, exaggarated, and unnatural as in any comic book.

The visuals, then. If you are a fan of over-the-top carnivalesque stages and dresses and, ahem, state-of-the-art CGI that has to make up for a lack of true cinematography, then this movie is for you. I know it is an action movie, but does that really mean that the makers need to test the audience's susceptibility to epilepsy? It is a poor man's shock & awe to the senses.

In all, we sat the show out, had an ice-cream, then moved on to wider horizons.

At home, I checked the IMDB score and couldn't believe my eyes. A 7.9 for this mess of a movie? I can understand temporary 'popularity', but this is ridiculous. I don't fall easily for conspiracy theories and scores being 'bought' by companies, but there is no explaining a 7.9 for 'Black Panther'.

Next, I looked at IMDB's user reviews. I noticed that they were much less favorable than the 'user ratings' indicate. No really, there is a HUGE gap between 'user ratings' and 'user review' ratings. How can that be explained?

So I checked the last nearly 300 reviews. What did I find?

People who took the trouble to write a few paragraphs about this movie, gave an average 4.3 to 'Black Panther'. A 4.3! That is nearly half of the rating the movie supposedly received from IMDB users in general. This is a ridiculous difference. IMDB may have its own algorithm to arrive at some 'weighted average', but that cannot account for this huge gap.

Here is a breakdown of the ratings awarded by 'IMDB user reviewers': 10 - 5,5% 9 - 4.5% 8 - 3.4% 7 - 8.3% 6 - 16.6% 5 - 15.5% 4 - 10.3% 3 - 8.3% 2 - 8.6% 1 - 19%

Please note that only 13.4% of IMDB user reviewers awarded 'Black Panther' with an 8, 9 or 10. While according to IMDB, no less than 66.4% of 'IMDB users' have given these ratings.

Sorry, this cannot be rationally explained, unless one makes untenable assumptions about the difference between 'users' and 'reviewers'.

In conclusion: all I can offer is my personal opinion about this movie. I am sorry that I invested time and money into seeing it. Neither my kid of 12 enjoyed it, neither did I. We had to choose between Disney's 'Coco' and 'Black Panther'. I think I'll make up for the disappointment of BP by taking my kid to 'Coco'.
'''

r2 = '''
This film is amazing watched in imax 3d. I watch most comic book movies this way for the last ten years,black panther is on another level for a standalone film the action affects story pacing brilliant never bored once. I wanted to watch it straight away again the only better marvel movie is the original avengers and logan but that isn't really marvel. Trust me don't listen to all the haters and watch this film I'm going to watch it again and I haven't done that with a film since the matrix enough said long live the KING.
'''

docs_new = [r1, r2]
predictions_b = bayesian_model.predict(docs_new)
predictions_svm = svm_model.predict(docs_new)

print("**** Bayes *****\n")
for doc, category in zip(docs_new, predictions_b):
  print(f"{doc[:100]} => {category}")

print("**** SVM *****\n")
for doc, category in zip(docs_new, predictions_svm):
  print(f"{doc[:100]} => {category}")
