# The following are the parameters that are used in the paper for both the transformers
EMBEDDING_DIM = 768 # same for hidden dim - same var is used
NUM_HEADS = 12 # same for self and multi-head attention
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.3
MAX_LEN = 300
FF_DIM = 3072 # feed forward dim = 4*embedding dim <= apparently same was used in gpt2-117m
NUM_CONNECTORS = 8 # number of discourse relation classes
LAMBDA1 = 0.1
LAMBDA2 = 0.3

# The following are the parameters that are used arbitrarity
NUM_DECODERS = 1
NUM_EPOCHS = 200
BATCH_SIZE = 2 # change it to 8
THRESHOLD = 0.1

DISCOURSE_MARKERS = [
    'and', 'but', 'because', 'when', 'if', 'so', 'before', 'though'
]