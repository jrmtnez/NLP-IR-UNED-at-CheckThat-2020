# NLP&IR@UNED at CheckThat! 2020, Task 1 English

Source code used for NLP&IR@UNED team participation in the English task 1 of the Checkthat! 2020 Lab (https://sites.google.com/view/clef2020-checkthat).

To export submission files for *primary*, *contrastive1* and *contrastive2* runs:

    python nlpir01/task1.py

Exported files are stored in *nlpir01/results* folder. The English version of task 1, unlike the Arabic version, is calculated simultaneously on the traninig and test datasets.

Note that results may vary due to random parameter initialization.

To run all combinations of *embeddings* and *graph*:

    python nlpir01/task1.py -ca 1

To see other options:

    python nlpir01/task1.py -h

## Runs

### primary

runid: T1-EN-nlpir01-run4

Description: Bidirectional LSTM model with 10 tanh units in the hidden layers, Glove English embeddings, number of inputs increased using a graph to locate related tweets.

### contrastive 1

runid: T1-EN-nlpir01-run1

Description: FFNN with 1000 hard_sigmoid units in the only hidden layer, Glove English embeddings, number of inputs increased using a graph to locate related tweets.


### contrastive 2

runid: T1-EN-nlpir01-run2

Description: CNN with 32 sigmoid units in the two convolutional layers, Glove English embeddings, number of inputs increased using a graph to locate related tweets.

## Installation instructions

- Clone official repository of English task 1 (https://github.com/sshaar/clef2020-factchecking-task1).
- Copy folder *nlpir01* into the downloaded repository.
- *nlpir01/resources* folder must contain a *glove_twitter_27B_200d.txt* file (http://nlp.stanford.edu/data/glove.twitter.27B.zip).
