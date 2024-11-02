import re
import torch
import nltk
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from params import *
from temp_discourse_model import StoryTransformer

class DiscourseTextDataset(Dataset):
    def __init__(self, data, tokenizer, word2index, index2word, train=False):
        self.data = data
        self.isTrain = train
        self.tokenizer = tokenizer
        self.word2index = word2index
        self.index2word = index2word

    def get_sentence_boundaries(self, text):
        """Get start and end indices of sentences in the text"""
        sentences = sent_tokenize(text)
        boundaries = []
        current_idx = 0
        
        for sentence in sentences:
            # Get the token count for this sentence
            tokens = self.tokenizer.encode(sentence)
            start_idx = current_idx
            end_idx = current_idx + len(tokens)
            boundaries.append((start_idx, end_idx))
            current_idx = end_idx
            
        return boundaries

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, outline, story = (
            self.data[idx]['prompt'], 
            self.data[idx]['outline'], 
            self.data[idx]['story']
        )

        input_sequence = prompt + " <s> " + outline + " <sep> "
        if self.isTrain:
            input_sequence += story

        # Get token indices
        input_sequence_indices = self.tokenizer.encode(
            input_sequence, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_LEN, 
            return_tensors='pt'
        ).squeeze()

        if story:
            story_indices = self.tokenizer.encode(
                story,
                truncation=True,
                padding='max_length',
                max_length=MAX_LEN,
                return_tensors='pt'
            ).squeeze()
            # Get sentence boundaries for the story
            sentence_boundaries = self.get_sentence_boundaries(story)
        else:
            pad_index = self.word2index['<pad>']
            story_indices = torch.full((MAX_LEN,), pad_index, dtype=torch.long)
            sentence_boundaries = []

        # Simulate discourse labels (replace this with actual discourse classifier output)
        if len(sentence_boundaries) > 1:
            discourse_labels = torch.randint(0, 9, (len(sentence_boundaries)-1,))
        else:
            discourse_labels = torch.tensor([])

        return input_sequence_indices, story_indices, sentence_boundaries, discourse_labels

def get_discourse_data_loader(tokenized_data, tokenizer, model, train=True):
    dataset = DiscourseTextDataset(
        tokenized_data, 
        tokenizer, 
        model.word_to_index, 
        model.index_to_word, 
        train
    )
    return DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_discourse_batch
    )

def collate_discourse_batch(batch):
    """Custom collate function to handle variable-length sentence boundaries and discourse labels"""
    input_seqs, target_seqs, sentence_boundaries, discourse_labels = zip(*batch)
    
    # Pad input and target sequences
    input_seqs = torch.stack(input_seqs)
    target_seqs = torch.stack(target_seqs)
    
    # Convert boundaries and labels to padded tensors
    max_sentences = max(len(bounds) for bounds in sentence_boundaries)
    max_labels = max(len(labels) for labels in discourse_labels)
    
    # Pad sentence boundaries
    padded_boundaries = []
    for bounds in sentence_boundaries:
        if len(bounds) < max_sentences:
            bounds = bounds + [(0, 0)] * (max_sentences - len(bounds))
        padded_boundaries.append(bounds)
    
    # Pad discourse labels
    padded_labels = []
    for labels in discourse_labels:
        if len(labels) < max_labels:
            labels = torch.cat([labels, torch.full((max_labels - len(labels),), -1)])
        padded_labels.append(labels)
    
    padded_labels = torch.stack(padded_labels) if padded_labels else torch.tensor([])
    
    return input_seqs, target_seqs, padded_boundaries, padded_labels

import torch
import json
from torch.nn import functional as F

def dataloader_helper(data_path, target_path, start_idx, chunk_size=1000):
    """
    Load data from files in chunks for memory efficiency
    
    Args:
        data_path: Path to input data file
        target_path: Path to target data file 
        start_idx: Starting index for chunk
        chunk_size: Number of examples to load at once
    
    Returns:
        List of dictionaries containing prompt, outline and story
    """
    processed_data = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        # Skip to start_idx
        for _ in range(start_idx):
            next(f)
            
        # Read chunk_size examples
        count = 0
        for line in f:
            if count >= chunk_size:
                break
                
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                prompt = data.get('prompt', '')
                outline = data.get('outline', '')
                
                # Get corresponding target from target file
                with open(target_path, 'r', encoding='utf-8') as tf:
                    for _ in range(start_idx + count):
                        next(tf)
                    target = next(tf).strip()
                
                processed_data.append({
                    'prompt': prompt,
                    'outline': outline,
                    'story': target
                })
                
                count += 1
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON at line {start_idx + count}")
            except Exception as e:
                print(f"Error processing line {start_idx + count}: {str(e)}")
                
    return processed_data

def compute_loss(lm_output=None, targets=None, discourse_output=None, discourse_labels=None, lambda1=0.1):
    """
    Compute combined loss for language modeling and discourse classification
    
    Args:
        lm_output: Language model logits (B, S, V)
        targets: Target indices (B, S)
        discourse_output: Discourse classification logits (B, N, C) 
        discourse_labels: Discourse labels (B, N)
        lambda1: Weight for discourse loss
        
    Returns:
        Combined loss value
    """
    # Initialize total loss
    total_loss = 0
    
    # Language modeling loss
    if lm_output is not None and targets is not None:
        # Reshape outputs and targets for loss calculation
        batch_size, seq_len, vocab_size = lm_output.size()
        lm_output = lm_output.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # Calculate cross entropy loss ignoring padding tokens
        padding_idx = targets == 0  # Assuming 0 is padding index
        valid_targets = targets[~padding_idx]
        valid_outputs = lm_output[~padding_idx]
        
        if len(valid_targets) > 0:
            lm_loss = F.cross_entropy(valid_outputs, valid_targets)
            total_loss += lm_loss

    # Discourse classification loss
    if discourse_output is not None and discourse_labels is not None:
        # Only calculate loss for valid labels (-1 indicates padding)
        valid_mask = discourse_labels != -1
        valid_discourse_output = discourse_output[valid_mask]
        valid_discourse_labels = discourse_labels[valid_mask]
        
        if len(valid_discourse_labels) > 0:
            discourse_loss = F.cross_entropy(valid_discourse_output, valid_discourse_labels)
            total_loss += lambda1 * discourse_loss
            
    return total_loss

def init_weights(model):
    """
    Initialize model weights
    
    Args:
        model: PyTorch model
    """
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

def train_discourse(model, train_loader, optimizer, device, loss_function, lambda1=0.1):
    model.train()
    total_loss = 0
    
    for input_seq, target_seq, sentence_boundaries, discourse_labels in train_loader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        discourse_labels = discourse_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        lm_output, discourse_output = model(
            input_seq, 
            target_seq, 
            sentence_boundaries, 
            discourse_labels
        )
        
        # Calculate combined loss
        loss = compute_loss(
            lm_output=lm_output,
            targets=target_seq,
            discourse_output=discourse_output,
            discourse_labels=discourse_labels,
            lambda1=lambda1
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_discourse(model, loader, device, loss_function, lambda1=0.1):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_seq, target_seq, sentence_boundaries, discourse_labels in loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            discourse_labels = discourse_labels.to(device)
            
            # Forward pass
            lm_output, discourse_output = model(
                input_seq, 
                target_seq, 
                sentence_boundaries, 
                discourse_labels
            )
            
            # Calculate combined loss
            loss = compute_loss(
                lm_output=lm_output,
                targets=target_seq,
                discourse_output=discourse_output,
                discourse_labels=discourse_labels,
                lambda1=lambda1
            )
            
            total_loss += loss.item()
            
    return total_loss / len(loader)

def generate_story_with_discourse(model, prompt, outline, tokenizer, device, max_length=MAX_LEN):
    model.eval()
    
    # Prepare input sequence
    input_sequence = prompt + " <s> " + outline + " <sep> "
    input_indices = tokenizer.encode(
        input_sequence, 
        truncation=True, 
        padding='max_length', 
        max_length=max_length, 
        return_tensors='pt'
    ).to(device)
    
    # Initialize target sequence
    target_indices = torch.full(
        (1, 1), 
        tokenizer.bos_token_id, 
        dtype=torch.long, 
        device=device
    )
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            lm_output, _ = model(input_indices, target_indices)
            
            # Get next token probabilities
            next_token_logits = lm_output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append to target sequence
            target_indices = torch.cat([target_indices, next_token], dim=1)
            
            # Check for end of sequence
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated story
    generated_story = tokenizer.decode(target_indices[0], skip_special_tokens=True)
    return generated_story

def main():
    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the discourse-aware model
    model = StoryTransformer(tokenizer=tokenizer, device=device)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    with open("temp_train.txt", 'r') as fp:
        lines = len(fp.readlines())
    num_loops = (lines // (BATCH_SIZE*CHUNK_SIZE)) + 1

    for i in range(num_loops):
        train_data = dataloader_helper("temp_train.txt", "temp_train_target.txt", i * CHUNK_SIZE * BATCH_SIZE)
        train_loader = get_discourse_data_loader(train_data, tokenizer, model, True)
        
        print(f"Training on chunk {i}")
        for epoch in range(NUM_EPOCHS):
            train_loss = train_discourse(model, train_loader, optimizer, device, compute_loss)
            eval_loss = evaluate_discourse(model, train_loader, device, compute_loss)
            
            print(f"Epoch {epoch+1} train loss: {train_loss}")
            print(f"Epoch {epoch+1} eval loss: {eval_loss}")
    
    # Save the model
    torch.save(model.state_dict(), "discourse_transformer.pth")

if __name__ == "__main__":
    main()