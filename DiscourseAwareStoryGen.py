import torch
import torch.nn as nn
import torch.nn.functional as F
from params import DISCOURSE_MARKERS, MAX_LEN
from GoldenBert import predict_discourse_marker

class DiscourseAwareStoryGenerator(nn.Module):
    def __init__(self, encoder, hidden_size, output_size, tokenizer, device):
        super(DiscourseAwareStoryGenerator, self).__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.encoder = encoder
        self.device = device  # Ensure the device is stored
        self.Wf = nn.Linear(2 * hidden_size, hidden_size).to(device)
        self.bf = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).to(device))
        self.Wo = nn.Linear(hidden_size, output_size).to(device)
        self.bo = nn.Parameter(torch.zeros(output_size, dtype=torch.float32).to(device))

    def encode_sentence(self, sentence_tokens):
        encoded = self.tokenizer(sentence_tokens, return_tensors='pt', truncation=True, padding=True).to(self.device)
        outputs = self.encoder(**encoded)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        sentence_embedding = torch.mean(last_hidden_state, dim=1).squeeze()  # Mean pooling
        return sentence_embedding.to(self.device)

    def forward(self, sentences):
        sentence_reps = [self.encode_sentence(s) for s in sentences]
        hs_i = sentence_reps[0].to(self.device)
        hs_next = sentence_reps[1].to(self.device)
        concatenated = torch.cat([hs_i, hs_next], dim=0).to(self.device)  # Shape: (2 * hidden_size)
        f = torch.tanh(self.Wf(concatenated) + self.bf)  # Shape: (hidden_size)
        logits = self.Wo(f) + self.bo  # Shape: (output_size)
        prob = F.softmax(logits, dim=-1)  # Classification probabilities
        return prob

def train_step(model, sentences, golden_bert, golden_bert_tokenizer):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    prob = model(sentences).unsqueeze(0)

    marker, _ = predict_discourse_marker(golden_bert, golden_bert_tokenizer, sentences[0], sentences[1], model.device)
    true_marker_index = torch.tensor([DISCOURSE_MARKERS.index(marker)], dtype=torch.long).to(model.device)

    loss = criterion(prob, true_marker_index)
    optimizer.step()
    
    return loss
