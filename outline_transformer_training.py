import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import pandas as pd
from torch.utils.data import DataLoader
# from gensim.models import Word2Vec
import math
from statistics import mean

from tqdm import tqdm
import sys

from transformers import GPT2Tokenizer
from transformers import logging
logging.set_verbosity_error()

from get_outline import generate_outline
from outline_transformer import OutlineTransformer
from utils import TextDataset, read_text, create_datasets, collate_fn

print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU")
    print(torch.cuda.get_device_name())
    device = torch.device('cuda')
else:
    print("CPU")
    device = torch.device('cpu')

# Define some constants
embedding_dimension = 100
num_attention_heads = 4
num_decoders = 4
feed_forward_dim = 400
dropout_rate = 0.1
learning_rate = 0.001
batch_size = 2
num_epochs = 1

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.bos_token = '<start>'
tokenizer.eos_token = '<end>'

# window = 10

# word_vectors = Word2Vec(vector_size=embedding_dimension, window=5, min_count=1, workers=4)
# word_vectors.build_vocab(['<start>', '<end>'], update=False)

# for i in range(1):
#     training_input_prompts, training_targets = load_text("archive/writingPrompts/train.wp_source", "archive/writingPrompts/train.wp_target", 0, window)
#     val_input_prompts, val_targets = load_text("archive/writingPrompts/valid.wp_source", "archive/writingPrompts/valid.wp_target", 0, window)
#     print("Max: ", max(len(sentence) for sentence in training_input_prompts), "Mean: ", mean(len(sentence) for sentence in training_input_prompts), "\n")
#     print("Max: ", max(len(sentence) for sentence in training_targets), "Mean: ", mean(len(sentence) for sentence in training_targets), "\n")
#     print("Max: ", max(len(sentence) for sentence in val_input_prompts), "Mean: ", mean(len(sentence) for sentence in val_input_prompts), "\n")
#     print("Max: ", max(len(sentence) for sentence in val_targets), "Mean: ", mean(len(sentence) for sentence in val_targets), "\n")

#     print(training_input_prompts, "\n", training_targets)
    
#     word_vectors.build_vocab(training_input_prompts, update=True)

# embedding_matrix = word_vectors.wv
# print(len(embedding_matrix))

# sys.exit()

# Read the training, validation and testing prompts and generate their correspondign outlines
training_prompts = read_text("archive/writingPrompts/train.wp_source")
training_outlines = [generate_outline(prompt) for prompt in training_prompts]
validation_prompts = read_text("archive/writingPrompts/train.wp_source")
validation_outlines = [generate_outline(prompt) for prompt in validation_prompts]
test_prompts = read_text("archive/writingPrompts/train.wp_source")
test_outlines = [generate_outline(prompt) for prompt in test_prompts]

# Create datasets
training_inputs, training_targets = create_datasets(training_prompts, training_outlines)
training_dataset = TextDataset(training_inputs, training_targets)
validation_inputs, validation_targets = create_datasets(validation_prompts, validation_outlines)
validation_dataset = TextDataset(validation_inputs, validation_targets)
test_inputs, test_targets = create_datasets(test_prompts, test_outlines)
test_dataset = TextDataset(test_inputs, test_targets)

# Create dataloaders
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = OutlineTransformer(embedding_dimension, tokenizer, num_attention_heads, num_decoders, feed_forward_dim, dropout_rate, device)
if torch.cuda.is_available():
    model = model.to(device)
loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_words = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}', unit='batch') as pbar:
        for inputs, targets in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = model.tokenize_to_indices(targets).reshape(-1)
            loss = loss_function(outputs, targets)
            loss.backward()
            batch_loss = loss.item() * len(targets)
            batch_words = len(targets)
            batch_perplexity = np.exp(batch_loss / batch_words)
            total_loss += batch_loss
            total_words += batch_words
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
    avg_loss = total_loss / total_words
    perplexity = np.exp(avg_loss)
    print(f'Epoch {epoch+1}, Training Loss: {avg_loss}, Training Perplexity: {perplexity}')
    
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch+1} Validation', unit='batch') as pbar:
            for inputs, targets in val_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = model.tokenize_to_indices(targets).reshape(-1)
                loss = loss_function(outputs, targets)
                batch_loss = loss.item() * len(targets)
                batch_words = len(targets)
                batch_perplexity = np.exp(batch_loss / batch_words)
                total_loss += batch_loss
                total_words += batch_words
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                avg_loss = total_loss / total_words
    perplexity = np.exp(avg_loss)
    print(f'Epoch {epoch+1}, Validation Loss: {avg_loss} Validation Perplexity: {perplexity}')

model.eval()
total_loss = 0
total_words = 0
with torch.no_grad():
    with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
        for inputs, targets in test_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets = model.tokenize_to_indices(targets).reshape(-1)
            loss = loss_function(outputs, targets)
            batch_loss = loss.item() * len(targets)
            batch_words = len(targets)
            batch_perplexity = np.exp(batch_loss / batch_words)
            total_loss += batch_loss
            total_words += batch_words
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
            avg_loss = total_loss / total_words
perplexity = np.exp(avg_loss)
print(f'Testing Loss: {avg_loss} Testing Perplexity: {perplexity}')

torch.save(model, "outline_transformer.pt")