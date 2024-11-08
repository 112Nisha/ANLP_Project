import torch
import torch.nn as nn
import torch.nn.functional as F
from golden_BERT import predict_discourse_marker
from params import NUM_CONNECTORS, EMBEDDING_DIM, DISCOURSE_MARKERS
from transformers import BertTokenizer

class DiscourseAwareStoryGenerator(nn.Module):
    def __init__(self, encoder, hidden_size, output_size,tokenizer, device):
        super(DiscourseAwareStoryGenerator, self).__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.encoder = encoder
        self.device = device
        self.Wf = nn.Linear(2 * hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.Wo = nn.Linear(hidden_size, output_size)
        self.bo = nn.Parameter(torch.zeros(output_size, dtype=torch.float32))

    def encode_sentence(self, sentence_tokens):
        encoded = self.tokenizer(sentence_tokens, return_tensors='pt', truncation=True, padding=True)
        outputs = self.encoder(**encoded)
        last_hidden_state = outputs.last_hidden_state 
        sentence_embedding = torch.mean(last_hidden_state, dim=1).squeeze() 
        
        return sentence_embedding

    def forward(self, sentences):
        sentence_reps = [self.encode_sentence(s) for s in sentences]
        hs_i = sentence_reps[0]
        hs_next = sentence_reps[1]
        concatenated = torch.cat([hs_i, hs_next], dim=0)  # Shape: (2 * hidden_size)
        f = torch.tanh(self.Wf(concatenated) + self.bf)
        logits = self.Wo(f) + self.bo  # shape: (output_size)
        prob = F.softmax(logits, dim=-1)  # Apply softmax to get classification probabilities
        return prob

def train_step(model, sentences, golden_bert, golden_bert_tokenizer):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    prob = model(sentences).unsqueeze(0)

    marker, _ = predict_discourse_marker(golden_bert, golden_bert_tokenizer, sentences[0], sentences[1], model.device)
    true_marker_index = torch.tensor([DISCOURSE_MARKERS.index(marker)], dtype=torch.long).to(model.device)

    loss = criterion(prob, true_marker_index)
    loss.backward()
    optimizer.step()
    
    return loss
