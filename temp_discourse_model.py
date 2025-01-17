import math
import torch
import torch.nn as nn
from utils import causal_masking
from params import NUM_DECODERS, NUM_HEADS, EMBEDDING_DIM, FF_DIM, DROPOUT_RATE

def get_embedding_with_positional_encoding(embedding, context_size, device):
    batch_size, sequence_length, embedding_dim = embedding.size()

    position = torch.arange(0, context_size).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)).to(device)

    positional_encoding = torch.zeros(context_size, embedding_dim).to(device)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)

    positional_encoding = positional_encoding[:sequence_length].unsqueeze(0)
    positional_encoding = positional_encoding.expand(batch_size, -1, -1)

    return embedding.to(device) + positional_encoding

class StoryTransformer(nn.Module):
    def __init__(self, tokenizer, device, embedding_dimension=EMBEDDING_DIM, num_attention_heads=NUM_HEADS, num_decoders=NUM_DECODERS, feed_forward_dim=FF_DIM, dropout_rate=DROPOUT_RATE):
        super(StoryTransformer, self).__init__()
        self.device = device
        self.embedding_dimension = embedding_dimension
        self.tokenizer = tokenizer

        self.word_to_index = tokenizer.get_vocab()
        vocab_size = len(self.word_to_index)
        self.word_to_index['<unk>'] = vocab_size
        self.word_to_index['<pad>'] = vocab_size + 1
        vocab_size = len(self.word_to_index)
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.tokenizer.vocab = self.word_to_index
        self.positional_encoding = get_embedding_with_positional_encoding
        decoder_block = nn.TransformerDecoderLayer(d_model=embedding_dimension, nhead=num_attention_heads, dim_feedforward=feed_forward_dim, dropout=dropout_rate, batch_first=True, device=device)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_block, num_layers=num_decoders)
        self.hidden_layer = nn.Linear(embedding_dimension, vocab_size) # hidden layer dim = embedding dim
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_seq, target):
        input_seq = self.embedding(input_seq)
        input_seq = self.dropout(input_seq)
        input_seq = self.positional_encoding(input_seq, self.embedding_dimension, self.device)

        target = self.embedding(target)
        target = self.dropout(target)
        target = self.positional_encoding(target, self.embedding_dimension, self.device)

        target_mask = causal_masking(target.size(1)).to(self.device)

        output = self.decoder(target, input_seq, tgt_mask=target_mask)
        output = self.dropout(output)
        output = self.hidden_layer(output)
        return output
