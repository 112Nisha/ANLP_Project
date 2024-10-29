import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import nltk
import re

# from get_outline import generate_outline

# def clean_training_data()

def get_nth_line(file_path, n):
    """
    Get the nth line from the file at file path.
    
    :param file_path: str, Path to the file.
    :param n: int, Line number to retrieve (1-based index).
    :return: str, The nth line from the file or None if line doesn't exist.
    """
    with open(file_path, 'r') as file:
        for current_line_number, line in enumerate(file, start=1):
            if current_line_number == n:
                return line.strip()
    return None

def get_nth_line_from_file(file, n):
    """
    Get the nth line from the file.
    
    :param file_path: str, Path to the file.
    :param n: int, Line number to retrieve (1-based index).
    :return: str, The nth line from the file or None if line doesn't exist.
    """
    for current_line_number, line in enumerate(file, start=1):
        if current_line_number == n:
            return line.strip()
    return None

# def load_text_and_model(embedding_dimension, prompt_file_path, story_file_path, model=True, second_tr=False):
#     if model:
#         model = Word2Vec(vector_size=embedding_dimension, window=5, min_count=1, workers=4)
    
#     inputs = []
#     targets = []
#     with open(prompt_file_path) as prompt_file:
#         with open(story_file_path) as story_file:
#             for i in range(10):
#                 prompt_file.seek(0)
#                 story_file.seek(0)
#                 prompt = get_nth_line_from_file(prompt_file, i)
#                 story = get_nth_line_from_file(story_file, i)
#                 outline = generate_outline(story)
#                 if prompt is not None and outline is not None:
#                     tokenized_sentence = tokenize(prompt)

#                     if model:
#                         if i == 0:
#                             model.build_vocab([tokenized_sentence], update=False)
#                         else:
#                             model.build_vocab([tokenized_sentence], update=True)
#                         model.train([tokenized_sentence], total_examples=1, epochs=1)

#                     input_prompt = '<start>' + prompt + '<sep>' + outline
#                     if second_tr:
#                         input_prompt = '<start>' + prompt + '<sep>' + outline + '<sep>' + story
#                     inputs.append(input_prompt)
#                     output_prompt = prompt + '<sep>' + outline + '<end>'
#                     if second_tr:
#                         output_prompt = prompt + '<sep>' + outline + '<sep>' + story + '<end>'
#                 targets.append(output_prompt)
#     if model:
#         return model, inputs, targets
#     else:
#         return inputs, targets

class TextDataset(Dataset):
    def __init__(self, input_prompts, targets):
        self.input_prompts = input_prompts
        self.targets = targets

    def __len__(self):
        return len(self.input_prompts)

    def __getitem__(self, idx):
        return (self.input_prompts[idx], self.targets[idx])

def get_sinusoidal_positional_encoding(input_embeddings):
    max_sentence_length = len(input_embeddings[0])
    embedding_dim = len(input_embeddings[0][0])
    encoded = torch.zeros(max_sentence_length, embedding_dim)
    position = torch.arange(0, max_sentence_length, dtype=torch.float).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-1) * torch.log(torch.tensor(10000.0) / embedding_dim))
    encoded[:, 0::2] = torch.sin(position * divisor)
    encoded[:, 1::2] = torch.cos(position * divisor)
    return input_embeddings + encoded[:input_embeddings.shape[1], :].to(input_embeddings.device)

# def get_rotational_positional_encoding(self, sequence_length, embedding_dimension):
    
def causal_masking(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)
    mask = torch.full_like(mask, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

def tokenize(input):
    output = []
    for prompt in input:
        prompt = re.sub(r'([.,!?;:*()"\'“”‘’_\u2014-])', r' \1 ', prompt)
        prompt = re.sub(r'[^\w\s]', '', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = prompt.strip()
        prompt = prompt.lower()
        prompt = nltk.tokenize.word_tokenize(prompt)
        output.append(prompt)
    return output

# def word2vec_embeddings(model, text):
    # if model is None:
    #     model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    

# def collate_fn(batch):
