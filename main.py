# Read data from the folders


import argparse
import logging
import random
from data_utils import DataLoader

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
    args.file, 'train', limit=int(args.num_records))

# show samples for debugging
for i in range(args.num_records):
  logging.info(f"sample review {training_data.data[i][:200]}, ratings {training_data.ratings[i]}, sentiment {training_data.sentiments[i]}")
