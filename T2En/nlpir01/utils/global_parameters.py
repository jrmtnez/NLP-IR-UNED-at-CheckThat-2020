TWEETS_TRAIN_PATH = "data/v3.0/train/tweets.queries.tsv"
GOLD_LABELS_TRAIN_PATH = "data/v3.0/train/tweet-vclaim-pairs.qrels"
TWEETS_DEV_PATH = "data/v3.0/dev/tweets.queries.tsv"
GOLD_LABELS_DEV_PATH = "data/v3.0/dev/tweet-vclaim-pairs.qrels"

TWEETS_TEST_PATH = "test-input/test-input/tweets.queries.tsv"

VCLAIMS_PATH = "data/v3.0/verified_claims.docs.tsv"
RESULTS_FILE_PATH = "nlpir01/results/nlpir01.results"

TRAIN_DF_FILE = "nlpir01/data/train_df.pkl"
DEV_DF_FILE = "nlpir01/data/dev_df.pkl"
TEST_DF_FILE = "nlpir01/data/test_df.pkl"
TRAIN_JSON_FILE = "nlpir01/data/train.json"
DEV_JSON_FILE = "nlpir01/data/dev.json"
TEST_JSON_FILE = "nlpir01/data/test.json"

PREDICT_FILE = "nlpir01/results/nlpir01.predictions"
RESULT_FILE = "nlpir01/results/nlpir01.result"
PREDICT_SUB_FILE = "nlpir01/results/T2-EN-nlpir01-run_type.txt"
TRAINING_PNG_FILE = "nlpir01/results/nlpir01.training_acc_loss.png"

PREDICT_FILE_COLUMNS = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag']

LABEL_COLUMN = "label"

NUM_WORDS = 15000
EMBEDDING_SIZE = 200

BATCH_SIZE = 32
EPOCHS = 100
