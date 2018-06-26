README

This program looks at a set of movie reviews from ai.stanford.edu/~amaas/data/sentiment/ and tries to make a ML model that can be used to classify sentiments in the review.

In the future, we will implement some ideas from the paper http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf to capture  semantic similarities among words using a probablistic model. This is similar to LDA


## Packages to install
numpy, pandas, scipy, matplotlib


## Steps to run
1. Extract the aclImdb_v1.tar.gz to a directory. This contains dataset for movie reviews
1. In the root directory, execute the code
    python main.py -f <extracted_dir_path> 
..* For eg: python  main.py  -f ./aclImdb/  -v -n 100
1. For actual usage - run python main.py --review 'review text' where review text 

## Architecture 


## Steps to do 
1. Load the training set.
1. (abdul) Bar total positive and negative reviews
1. (abdul) Simple ?? in matplot average length of movie review, negative and positive review (we have 3 values - how to show ?)

1. Run a bayesian model
1. Show reporting accuracy
(abdul) Plot for a sample of 100 test reviews, our model calculation vs actual

1. Run a SVD
1. Show reporting accuracy

1. Run CV
1. Show report
1.(abdul) Plot accuracy of each CV.

1. Run GBT
2. Show report
3. (abdu) plot accuracy of model


## Future
Use ratings category instead of boolean for classification







