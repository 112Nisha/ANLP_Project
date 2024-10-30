import math
import torch
import torch.nn as nn
from utils import causal_masking

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
    def __init__(self, embedding_dimension, embedding_matrix, num_attention_heads, num_decoders, feed_forward_dim, dropout_rate, device):
        super(StoryTransformer, self).__init__()
        self.device = device
        existing_vectors = torch.tensor(embedding_matrix.vectors)
        num_existing_vectors = existing_vectors.size(0)
        adjusted_embedding_matrix = torch.zeros((num_existing_vectors + 1, embedding_dimension))
        adjusted_embedding_matrix[:num_existing_vectors] = existing_vectors
        adjusted_embedding_matrix[-1] = torch.zeros(embedding_dimension)
        self.embedding = nn.Embedding.from_pretrained(adjusted_embedding_matrix, freeze=False)
        self.embedding_dimension = embedding_dimension

        self.word_to_index = {word: idx for idx, word in enumerate(embedding_matrix.key_to_index)}
        self.word_to_index['<unk>'] = num_existing_vectors
        self.index_to_word = list(embedding_matrix.index_to_key) + ['<unk>']

        self.positional_encoding = get_embedding_with_positional_encoding
        decoder_block = nn.TransformerDecoderLayer(embedding_dimension, num_attention_heads, feed_forward_dim, dropout_rate, batch_first=True, device=device)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_block, num_layers=num_decoders)
        self.hidden_layer = nn.Linear(embedding_dimension, embedding_dimension)
    
    def forward(self, input_seq, target):
        input_seq = self.embedding(input_seq)
        input_seq = self.positional_encoding(input_seq, self.embedding_dimension, self.device)

        target = self.embedding(target)
        target = self.positional_encoding(target, self.embedding_dimension, self.device)

        target_mask = causal_masking(target.size(1)).to(self.device)

        output = self.decoder(target, input_seq, tgt_mask=target_mask)
        output = self.hidden_layer(output)
        return output
