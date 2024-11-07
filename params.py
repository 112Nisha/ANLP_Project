
# The following are the parameters that are used in the paper for both the transformers
EMBEDDING_DIM = 768 # same for hidden dim - same var is used
NUM_HEADS = 12 # same for self and multi-head attention
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.3
MAX_LEN = 500
FF_DIM = 3072 # feed forward dim = 4*embedding dim <= apparently same was used in gpt2-117m
NUM_CONNECTORS = 9 # number of discourse relation classes

# The following are the parameters that are used arbitrarity
NUM_DECODERS = 1
NUM_EPOCHS = 5
BATCH_SIZE = 2 # change it to 8
CHUNK_SIZE = 3 # change it to 1000