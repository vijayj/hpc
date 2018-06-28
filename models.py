
##################
# This class helps abstracting the various models and provides a simple interface
# to build a model, predict from the model, and analyze model's efficiency
##################
# Run the bayesian classifier on training data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn import metrics


class Model(object):

  def __init__(self, name):
    self.name = name
    self._create()

  def _create(self):
    if self.name == 'bayes':
      # create a classifier
      self.classifier = Pipeline([
          ('vect', CountVectorizer()),
          ('tfidf', TfidfTransformer()),
          ('clf', MultinomialNB())
      ])
    elif self.name == 'svm':
      self.classifier = Pipeline([
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
    else:
      raise ValueError(
          "Invalid model.Cannot create model for :{}".format(self.name))

  def train(self, X, Y):
    # Train the bayesian_model
    self.model = self.classifier.fit(
        X, Y)

  def predict(self, X):
    return self.model.predict(X)

  def analysis(self, actual, predicted):
    print("Accurancy ({0} model): {1}".format(self.name, np.mean(actual ==
                                                                 predicted)))

    if self.name == 'svm':
      print(metrics.classification_report(actual,
                                          predicted,
                                          target_names=['True', 'False']))

      # Show the line plot here
