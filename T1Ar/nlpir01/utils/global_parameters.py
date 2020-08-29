RESOURCES_PATH = "nlpir01/resources"
RESULTS_PATH = "nlpir01/results"
RESULTS_TO_EVALUATE_PATH = "nlpir01/results/pred"
GOLD_PATH = "nlpir01/results/gold"

TWEETS_PATH = "data/2020/task1/training/CT20-AR-Train-T1-Tweets.gz"
LABELS_PATH = "data/2020/task1/training/CT20-AR-Train-T1-Labels.txt"
TOPICS_PATH = "data/2020/task1/training/CT20-AR-Train-Topics.json"

TEST_TWEETS_PATH = "data/2020/task1/testing/CT20-AR-Test-T1-Tweets.gz"
TEST_TOPICS_PATH = "data/2020/task1/testing/CT20-AR-Test-Topics.json"

EMBEDDINGS_FILE = "vectors_256d_ar.txt"
RESULTS_FILE_PREFIX = "T1-AR-"

RESULTS_PER_CLAIM = 500

TEXT_COLUMN = "text"
TEXT_COLUMN2 = "claim_text"
LABEL_COLUMN = "label"

SEQ_LEN = 25
CLAIM_SEQ_LEN = 100

NUM_WORDS = 15000
EMBEDDING_SIZE = 256

BATCH_SIZE = 32
EPOCHS = 200
