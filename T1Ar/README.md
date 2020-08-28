# NLP&IR@UNED at CheckThat! 2020, Task 1 Arabic

Source code used for NLP&IR@UNED team participation in the Arabic task 1 of the Checkthat! 2020 Lab (https://sites.google.com/view/clef2020-checkthat).

*scorer/main.py* is an adapted version of the file in English task 1 (https://github.com/sshaar/clef2020-factchecking-task1).

To export submission files for primary, contrastive1 and contrastive2 runs:

    python nlpir01/task1.py -s 1 -g 1

Exported files are stored in *nlpir01/results* folder.

Note that results may vary due to random parameter initialization.

To run on *training* and *dev* datasets with different configurations and verbose results:

    python nlpir01/task1.py

To see other options:

    python nlpir01/task1.py -h

## Runs

### primary

runid: T1-AR-nlpir01-run1

Description: FFNN with 2000 relu units in the only hidden layer, Glove Arabic embeddings, number of inputs increased using a graph to locate related tweets.

### contrastive 1

runid: T1-AR-nlpir01-run2

Description: CNN with 100 relu units in the two convolutional layers, Glove Arabic embeddings, number of inputs increased using a graph to locate related tweets.

### contrastive 2

runid: T1-AR-nlpir01-run4

Description: Bidirectional LSTM model with 10 tanh units in the hidden layers, Glove Arabic embeddings, number of inputs increased using a graph to locate related tweets.

## Installation instructions

- *training* and *testing* folders in *data/2020/task1* must contain the datasets released by the organization (https://gitlab.com/bigirqu/checkthat-ar/-/tree/master/data/2020/task1).
- *nlpir01/resources* folder must contain a *vectors_256d_ar.txt* file (https://archive.org/download/arabic_corpus/vectors.txt.xz).
