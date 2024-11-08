import torch
import torch.nn as nn

from utils import get_sinusoidal_positional_encoding, causal_masking, tokenize

class OutlineTransformer(nn.Module):
    def __init__(self, embedding_dimension, tokenizer, num_attention_heads, num_decoders, feed_forward_dim, dropout_rate, device):
        super(OutlineTransformer, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_dimension)
        self.positional_encoding = get_sinusoidal_positional_encoding
        decoder_block = nn.TransformerDecoderLayer(embedding_dimension, num_attention_heads, feed_forward_dim, dropout_rate, batch_first=True, device=device)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_block, num_layers=num_decoders)
        self.hidden_layer = nn.Linear(embedding_dimension, tokenizer.vocab_size)
    
    def tokenize(self, target):
        return self.tokenizer(target)
    
    def forward(self, prompt):
        # Determine mode of operation and convert input as necessary
        inference = False
        if isinstance(prompt, str):
            prompt = [prompt]
            inference = True

        # Tokenize prompt
        prompt_indices = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Get prompt embeddings
        prompt_embeddings = self.embedding(prompt_indices)

        # # Convert prompt into indices keeping batching in mind
        # prompt_indices = []
        # for batch in prompt:
        #     indices = []
        #     for word in batch:
        #         if word in self.word_to_index:
        #             indices.append(self.word_to_index[word])
        #         else:
        #             indices.append(self.word_to_index['<unk>'])
        #     prompt_indices.append(indices)
        # prompt_indices = torch.tensor(prompt_indices)
        # print(prompt_indices.shape)

        # # Convert prompt indices into embeddings
        # prompt_embeddings = self.embedding(prompt_indices)

        # Add positional encoding to prompt embeddings
        prompt_embeddings = self.positional_encoding(prompt_embeddings)

        # Generate causal mask
        mask = causal_masking(prompt_indices.shape[1]).to(self.device)

        output = self.decoder(tgt=prompt_embeddings, memory=prompt_embeddings, tgt_mask=mask)

        output = self.hidden_layer(output)

        # if inference:
        #     # Return last word

        return output