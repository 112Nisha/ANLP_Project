import torch
import torch.nn as nn
from utils import get_sinusoidal_positional_encoding, causal_masking

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

        self.word_to_index = {word: idx for idx, word in enumerate(embedding_matrix.key_to_index)}
        self.word_to_index['<unk>'] = num_existing_vectors
        self.index_to_word = list(embedding_matrix.index_to_key) + ['<unk>']

        self.positional_encoding = get_sinusoidal_positional_encoding
        decoder_block = nn.TransformerDecoderLayer(embedding_dimension, num_attention_heads, feed_forward_dim, dropout_rate, batch_first=True, device=device)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_block, num_layers=num_decoders)
        self.hidden_layer = nn.Linear(embedding_dimension, num_existing_vectors + 1)
    
    def forward(self, prompt):
        prompt_indices = [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in prompt]
        prompt_indices = torch.tensor(prompt_indices, dtype=torch.long).to(self.device)  # Ensure indices are on the correct device
        prompt_indices = prompt_indices.unsqueeze(0)  # Add batch dimension if needed

        prompt_embeddings = self.embedding(prompt_indices)  # This should work without index error
        prompt_embeddings = self.positional_encoding(prompt_embeddings)
        mask = causal_masking(prompt_indices.shape[1]).to(self.device)

        output = self.decoder(tgt=prompt_embeddings, memory=prompt_embeddings, tgt_mask=mask)
        output = self.hidden_layer(output)

        return output
