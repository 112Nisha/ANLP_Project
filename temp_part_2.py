import re
import torch
from functools import partial
from gensim.models import Word2Vec
from transformers import GPT2Tokenizer
from utils import get_nth_line_from_file
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from temp_outline_transformer import StoryTransformer

MAX_LEN = 512 # change it 2048
BATCH_SIZE = 2 # change it to 8
CHUNK_SIZE = 3 # change it to 1000
EMBEDDING_DIM = 512
NUM_HEADS = 4
NUM_DECODERS = 1
FF_DIM = 300
DROPOUT_RATE = 0.1
NUM_EPOCHS = 5

# SHOULD WE USE DROPOUT AND A SCHEDULER TO REDUCE OVERFITTING?
# SHOULD WE BE USING THE GPT2 TOKENIZER? bECAUSE IT'S FOR SUMMARIZATION TASKS SPECIFICALLY.

def preprocess(text):
    if text is None:
        return ""
    cleaned_text = text.replace("<newline>", "")
    output = re.sub(r'([.,!?;:*()"\'“”‘’_\u2014-])', r' \1 ', cleaned_text)
    output = re.sub(r'[^\w\s]', '', output)
    output = re.sub(r'\s+', ' ', output)
    output = output.strip()
    output = output.lower()
    return output

def dataloader_helper(source_filename, target_filename, start_index):
    datalist = []
    for curr_index in range(CHUNK_SIZE * BATCH_SIZE):
        prompt, story = get_nth_line_from_file(source_filename, start_index + curr_index), get_nth_line_from_file(target_filename, start_index + curr_index)
        outline = prompt # CHANGE THIS LATER
        prompt, story, outline = preprocess(prompt), preprocess(story), preprocess(outline)
        input_dict = {'prompt': prompt, 'outline': outline, 'story': story}
        datalist.append(input_dict)
    return datalist

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, word2index, index2word, train=False):
        self.data = data
        self.isTrain = train
        self.tokenizer = tokenizer
        self.word2index = word2index
        self.index2word = index2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, outline, story = self.data[idx]['prompt'], self.data[idx]['outline'], self.data[idx]['story']

        # Concatenate input sequence
        input_sequence = prompt + " <s> " + outline + " <sep> "
        if self.isTrain:
            input_sequence += story

        input_sequence_indices = [self.word2index.get(word, self.word2index['<unk>']) for word in input_sequence.split()]
        story_indices = [self.word2index.get(word, self.word2index['<unk>']) for word in story.split()] if story else []
        return input_sequence_indices, story_indices

def collate_fn(batch):
    input_seqs, target_seqs = zip(*batch)
        
    # Pad smaller sequences and truncate larger sequences
    input_seqs_padded = [seq[:MAX_LEN] if len(seq) > MAX_LEN else seq + [0] * (MAX_LEN - len(seq)) for seq in input_seqs]
    target_seqs_padded = [seq[:MAX_LEN] if len(seq) > MAX_LEN else seq + [0] * (MAX_LEN - len(seq)) for seq in target_seqs]
    
    # Convert to tensors and move to the device
    input_seqs_padded = torch.tensor(input_seqs_padded)
    target_seqs_padded = torch.tensor(target_seqs_padded)
    
    return input_seqs_padded, target_seqs_padded

def get_data_loader(tokenized_data, tokenizer, vocab, train=True):
    dataset = TextDataset(tokenized_data, tokenizer, vocab.word_to_index, vocab.index_to_word, train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=partial(collate_fn))

def get_sentences(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    sentences = []
    for line in lines:
        line = line.strip()
        line = preprocess(line)
        words = line.split()
        sentences.append(words)
    return sentences

def train(model, train_loader, optimizer, device, loss_function):
    model.train()
    total_loss = 0
    for input_seq, target_seq in train_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        outputs = model(input_seq, target_seq)
        # print('------------')
        # print(outputs.shape)
        # print(input_seq.shape)
        # print(target_seq.shape)
        loss = loss_function(outputs, target_seq)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, loader, device,loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq, target_seq)
            loss = loss_function(outputs, target_seq)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # CHANGE FILE NAMES
    with open("temp_train.txt", 'r') as fp:
        lines = len(fp.readlines())
    num_chunks = lines // CHUNK_SIZE
    # print("Number of lines: ",lines)
    # print("Number of chunks: ",num_chunks)
    # Num of lines = 30, num_chunks is 10, batch size is 2, chunk_size = 3
    # 1 -> 0-5, 2 -> 6-11, 3 -> 12-17, 4 -> 18-23, 5 -> 24-29, 6 -> 30-36
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentences_train = get_sentences("temp_train.txt")
    sentences_target = get_sentences("temp_train_target.txt")
    sentences = sentences_train + sentences_target
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    word2vec_model = Word2Vec(sentences, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
    model = StoryTransformer(EMBEDDING_DIM, word2vec_model.wv, NUM_HEADS, NUM_DECODERS, FF_DIM, DROPOUT_RATE, device)
    model.to(device)
    # print(list(model.word_to_index.items())[1700:1800])
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(num_chunks):
        train_data = dataloader_helper("temp_train.txt", "temp_train_target.txt", i * CHUNK_SIZE * BATCH_SIZE)
        train_loader = get_data_loader(train_data, tokenizer, model,True)
        print(f"Training on chunk {i}")
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, train_loader, optimizer, device,loss_fn)
            # CHANGE THIS TO VALIDATION_LOADER
            eval_loss = evaluate(model, train_loader, device, loss_fn)
            print(f"Epoch {epoch+1} train loss: {train_loss}")
            print(f"Epoch {epoch+1} eval loss: {eval_loss}")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
