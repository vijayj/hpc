import os
import random
import math


# Bunch class is copied from scikit learn

class Bunch(dict):
  """Container object for datasets

  Dictionary-like object that exposes its keys as attributes.

  >>> b = Bunch(a=1, b=2)
  >>> b['b']
  2
  >>> b.b
  2
  >>> b.a = 3
  >>> b['a']
  3
  >>> b.c = 6
  >>> b['c']
  6

  """

  def __init__(self, **kwargs):
    super(Bunch, self).__init__(kwargs)

  def __setattr__(self, key, value):
    self[key] = value

  def __dir__(self):
    return self.keys()

  def __getattr__(self, key):
    try:
      return self[key]
    except KeyError:
      raise AttributeError(key)

  def __setstate__(self, state):
    # Bunch pickles generated with scikit-learn 0.16.* have an non
    # empty __dict__. This causes a surprising behaviour when
    # loading these pickles scikit-learn 0.17: reading bunch.key
    # uses __dict__ but assigning to bunch.key use __setattr__ and
    # only changes bunch['key']. More details can be found at:
    # https://github.com/scikit-learn/scikit-learn/issues/6196.
    # Overriding __setstate__ to be a noop has the effect of
    # ignoring the pickled __dict__
    pass


class DataLoader(object):

  def __init__(self, logger):
    self.logging = logger

  def load_data(self, directory, subdir, randomize=False, limit=-1):
    """ Reads data from a file in  a given directory and subdirectory

    Keyword arguments:
    limit: Has controls to limit how many files to read, default -1 to read all
    randomize: To help read reviews in random order, default False
    """

    half_limit = math.ceil(limit / 2)
    positive_reviews = self._read_reviews(
        directory, subdir, 'pos', randomize, half_limit)
    # Create a data frame or data bag

    negative_reviews = self._read_reviews(
        directory, subdir, 'neg', randomize, limit - half_limit)

    return self._merged(positive_reviews, negative_reviews)

  def _merged(self, dataset1, dataset2):
    dataset1.data.extend(dataset2.data)
    dataset1.ratings.extend(dataset2.ratings)
    dataset1.sentiments.extend(dataset2.sentiments)
    return dataset1

  def _read_reviews(self, directory, subdir, leafdir, randomize, limit):

    review_dir = os.path.join(directory, subdir, leafdir)
    try:
      list_of_files = os.listdir(review_dir)
    except FileNotFoundError as fnf:
      self.logging.info(
          "returning empty as directory path is invalid {}".format(review_dir))
      return Bunch()

    try:
      # shuffling the order of the files..
      if randomize:
        random.shuffle(list_of_files)

      # truncating files if limit is given
      if limit != -1:
        list_of_files = list_of_files[:limit]

      # we will read the files
      texts = []
      ratings = []
      sentiments = []

      for filename in list_of_files:
        path = os.path.join(review_dir, filename)
        self.logging.debug("filename to be processed {} ".format(path))
        with open(path) as f:
          text = f.read()
          self.logging.debug('reading file {}'.format(text[0:100]))
          texts.append(text)
          rating = self._get_ratings(filename)
          ratings.append(rating)
          if int(rating) <= 4:
            sentiments.append(False)
          elif int(rating) >= 7:
            sentiments.append(True)

      return Bunch(data=texts, ratings=ratings, sentiments=sentiments)
    except FileNotFoundError as fnf:
      self.logging.info(
          "failed to process file {}".format(filename))
      return Bunch()

  def _get_ratings(self, filename):
      # the file name is of the form 12260_10.txt
      # re.match('\w+_(\w+)\.txt','12260_10.txt')[1]
    return filename.split('_')[1].split('.')[0]
