ABS_PATH = ""

RESOURCES_PATH = ABS_PATH + "nlpir01/resources"
EMBEDDINGS_FILE = "glove_twitter_27B_200d.txt"
RESULTS_PATH = ABS_PATH + "nlpir01/results"
RESULTS_FILE_PREFIX = "T1-EN-"

TRAINING_TWEETS_PATH = ABS_PATH + "data/training_v2.json"
TRAINING_PATH = ABS_PATH + "data/training_v2.tsv"

DEV_TWEETS_PATH = ABS_PATH + "data/dev_v2.json"
DEV_PATH = ABS_PATH + "data/dev_v2.tsv"

TEST_TWEETS_PATH = ABS_PATH + "test-input/test-input.json"
TEST_PATH = ABS_PATH + "test-input/test-input.tsv"


RESULTS_PER_CLAIM = 0

TEXT_COLUMN = "tweet_text"
# -v1
# LABEL_COLUMN = "claim_worthiness"
# +v1
LABEL_COLUMN = "check_worthiness"

SEQ_LEN = 50

NUM_WORDS = 15000
EMBEDDING_SIZE = 200

BATCH_SIZE = 32
EPOCHS = 100
