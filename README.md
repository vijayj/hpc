README

This program looks at a set of movie reviews from ai.stanford.edu/~amaas/data/sentiment/ and tries to make a ML model that can be used to classify sentiments in the review.

In the future, we will implement some ideas from the paper http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf to capture  semantic similarities among words using a probablistic model. This is similar to LDA


## Packages to install
numpy, pandas, scipy, sklearn, matplotlib


## Steps to run
1. Extract the aclImdb_v1.tar.gz to a directory. This contains dataset for movie reviews
1. In the root directory, execute the code
    ``python main.py -f <extracted_dir_path> ``
..* For eg: ``python  main.py  -f ./aclImdb/  -v -n 100``
1. For actual usage - go to the adhoc_testing.py and add a review. Add it to docs_new array and then you can run the program again to see the output


## Future
Use ratings category instead of boolean for classification







