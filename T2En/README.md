# NLP&IR@UNED at CheckThat! 2020, Task 2 English

Source code used for NLP&IR@UNED team participation in the task 2 of the Checkthat! 2020 Lab (https://github.com/sshaar/clef2020-factchecking-task2).

To export submission files for *primary*, *contrastive1* and *contrastive2* runs:

    python nlpir01/task2.py
    python nlpir01/task2.py -tf 1
    python nlpir01/task2.py

Exported files are stored in *nlpir01/results/T2-EN-nlpir01-run_type.txt* folder.

Note that results may vary due to random parameter initialization.

To see other options:

    python nlpir01/task2.py -h

## Runs

### primary

runid: T2-EN-nlpir01-primary

Description: FFNN with 1000 elu units in the first hidden layer, 500 elu units in the second hidden layer and 250 elu units in the last hidden layer, Universal Sentence Encoder embeddings of tweet and claim, and as features, cosine similarity of embeddings, and difference of type token ratio, average word count, noun count, verb count, word diversity and tag diversity.

### contrastive 1

runid: T2-EN-nlpir01-contrastive1

Description: FFNN with 1000 elu units in the first hidden layer, 500 elu units in the second hidden layer and 250 elu units in the last hidden layer, Universal Sentence Encoder embeddings of tweet and claim, and tweet and title, and as features, cosine similarity of embeddings, and difference of type token ratio, average word count, noun count, verb count, word diversity and tag diversity.


### contrastive 2

runid: T2-EN-nlpir01-contrastive2

Same configuration as primary

## Installation instructions

- Clone official repository of English task 5 (https://github.com/sshaar/clef2020-factchecking-task2).
- Copy folder *nlpir01* into the downloaded repository.
