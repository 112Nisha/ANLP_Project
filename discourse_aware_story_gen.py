import torch
import torch.nn as nn
import torch.nn.functional as F
from params import NUM_CONNECTORS, EMBEDDING_DIM
from transformers import BertTokenizer

class DiscourseAwareStoryGenerator(nn.Module):
    def __init__(self, encoder, hidden_size, output_size):
        super(DiscourseAwareStoryGenerator, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.Wf = nn.Linear(2 * hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        self.Wo = nn.Linear(hidden_size, output_size)
        self.bo = nn.Parameter(torch.zeros(output_size))

    def encode_sentence(self, sentence_tokens):
        Hs_i = self.encoder(sentence_tokens)  # shape: (seq_len, hidden_size)
        hs_i = torch.max(Hs_i, dim=0).values  # shape: (hidden_size)
        return hs_i

    def forward(self, sentences):
        sentence_reps = [self.encode_sentence(s) for s in sentences]
        probabilities = []

        for i in range(len(sentence_reps) - 1):
            hs_i = sentence_reps[i]
            hs_next = sentence_reps[i + 1]
            f = torch.tanh(self.Wf(torch.cat([hs_i, hs_next]) + self.bf))  # shape: (hidden_size)
            logits = self.Wo(f) + self.bo  # shape: (output_size)
            prob = F.softmax(logits, dim=-1)  # Apply softmax to get classification probabilities
            probabilities.append(prob)

        return torch.stack(probabilities)  # shape: (len(sentences) - 1, output_size)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = DiscourseAwareStoryGenerator(encoder=tokenizer, hidden_size=EMBEDDING_DIM, output_size=NUM_CONNECTORS)
