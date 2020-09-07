# NLP&IR@UNED at CheckThat! 2020, Task 5 English

Source code used for NLP&IR@UNED team participation in the task 5 of the Checkthat! 2020 Lab (https://sites.google.com/view/clef2020-checkthat).

To export submission files for *primary*, *contrastive1* and *contrastive2* runs:

    python nlpir01/task5.py -m 7
    python nlpir01/task5.py -m 7
    python nlpir01/task5.py -m 7 -o 1

Exported files are stored in *nlpir01/results/submission_debates* folder.

Note that results may vary due to random parameter initialization.

To run all combinations of *embeddings* and *oversampling* (this can take a long time):

    python nlpir01/task5.py -ca 1

To see other options:

    python nlpir01/task5.py -h

## Runs

### primary

runid: T5-EN-nlpir01-primary

Description: Bidirectional LSTM model with 256 tanh units in the hidden layers and 6B-100D Glove English embeddings.


### contrastive 1

runid: T5-EN-nlpir01-contrastive1

Description: Bidirectional LSTM model with 256 tanh units in the hidden layers and 6B-100D Glove English embeddings (same configuration that primary).


### contrastive 2

runid: T5-EN-nlpir01-contrastive2

Description: Bidirectional LSTM model with 256 tanh units in the hidden layers, batch size of 128, 6B-100D Glove English embeddings and oversampling of the minority class.


## Installation instructions

- Clone official repository of English task 5 (https://github.com/sshaar/clef2020-factchecking-task5).
- Copy folder *nlpir01* into the downloaded repository.
- *nlpir01/resources* folder must contain a *glove.6B.100d.txt* file (http://nlp.stanford.edu/data/glove.6B.zip).


## References

### RID (Regressive Imagery Dictionary)
Colin Martindale. Romantic Progression: The Psychology of Literary History, Hemisphere, Washington, DC, 1975.
Colin Martindale. The clockwork muse: The predictability of artistic change. The clockwork muse: The predictability of artistic change. Basic Books, New York, NY, US, 1990. Pages: xiv, 411.

Dictionary file adapted from: https://rdrr.io/github/kbenoit/quanteda.dictionaries/