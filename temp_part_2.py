import torch
from temp_outline_transformer import OutlineTransformer
from torch.utils.data import Dataset, DataLoader
import re
# from utils import get_nth_line_from_file
from evaluate import get_nth_line_from_file
from transformers import GPT2Tokenizer
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
from functools import partial

MAX_LEN = 512 # change it 2048
BATCH_SIZE = 2 # change it to 8
CHUNK_SIZE = 3 # change it to 1000
EMBEDDING_DIM = 300
NUM_HEADS = 4
NUM_DECODERS = 1
FF_DIM = 300
DROPOUT_RATE = 0.1

class Vocabulary:
    def __init__(self,sentences):
        word2vec_model = Word2Vec(sentences, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
        embedding_matrix = word2vec_model.wv
        self.word2index = embedding_matrix.key_to_index
        self.index2word = embedding_matrix.index_to_key
        self.word2index['<unk>'] = len(self.word2index)
        self.index2word.append('<unk>')

def preprocess(text):
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
        prompt = get_nth_line_from_file(source_filename,start_index + curr_index)
        story = get_nth_line_from_file(target_filename,start_index + curr_index)
        # CHANGE THIS LATER
        outline = prompt
        prompt = preprocess(prompt)
        story = preprocess(story)
        outline = preprocess(outline)
        input_dict = {'prompt': prompt, 'outline': outline, 'story': story}
        datalist.append(input_dict)
    return datalist

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, word2index,index2word,train=False):
        self.data = data
        self.isTrain = train
        self.tokenizer = tokenizer
        self.word2index = word2index
        self.index2word = index2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]['prompt']
        outline = self.data[idx]['outline']
        story = self.data[idx]['story']

        # Concatenate input sequence
        input_sequence = prompt + " <s> " + outline + " <sep> "
        if self.isTrain:
            input_sequence += story
        
        # return torch.tensor(input_sequence),torch.tensor(story)
        # input_tensor = torch.tensor(self.tokenizer.encode(input_sequence))
        # target_tensor = torch.tensor(self.tokenizer.encode(story))
        # return input_tensor, target_tensor
        input_sequence_indices = [self.word2index[word] if word in self.word2index else self.word2index['<unk>'] for word in input_sequence]
        story_indices = [self.word2index[word] if word in self.word2index else self.word2index['<unk>'] for word in story]
        return torch.tensor(input_sequence_indices), torch.tensor(story_indices)
        # Encode input and target sequences using the tokenizer
        # input_tensor = torch.tensor(self.tokenizer.encode(input_sequence), dtype=torch.long)
        # target_tensor = torch.tensor(self.tokenizer.encode(story), dtype=torch.long)
         
        # decoded_input = torch.tensor(self.tokenizer.decode(input_tensor.tolist(), skip_special_tokens=True))
        # decoded_target = torch.tensor(self.tokenizer.decode(target_tensor.tolist(), skip_special_tokens=True))
        # return decoded_input, decoded_target
        # return input_tensor, target_tensor

def collate_fn(batch, tokenizer):
    input_seqs, target_seqs = zip(*batch) 
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
    return input_seqs_padded, target_seqs_padded



def get_data_loader(tokenized_data, tokenizer, vocab, train=True):
    dataset = TextDataset(tokenized_data, tokenizer, vocab.word_to_index,vocab.index_to_word,train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer))

# class TextDataset(Dataset):
#     def __init__(self, data, tokenizer, train=False):
#         self.data = data
#         self.isTrain = train
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # input_sequence = self.data[idx]['prompt'].tolist() + "<s>" + self.data[idx]['outline'].tolist() + "<sep>"
#         input_sequence = self.data[idx]['prompt'] + "<s>" + self.data[idx]['outline'] + "<sep>"
#         if self.isTrain:
#             input_sequence += self.data[idx]['story']
#         input_sequence = torch.tensor(self.tokenizer.encode(input_sequence))
#         return input_sequence[:MAX_LEN], self.data[idx]['story']

# def get_data_loader(tokenized_data, tokenizer,train=True):
#     dataset = TextDataset(tokenized_data, tokenizer,train)
#     return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def get_sentences(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    sentences = []
    for line in lines:
        line = line.strip()
        line = preprocess(line)
        sentences.append(line)
    return sentences

def train(model, train_loader, optimizer, device,loss_function):
    model.train()
    total_loss = 0
    countval = 0
    print("START")
    for input_seq, target_seq in train_loader:
        # print("IN")
        # print(input_seq)
        # input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        outputs = model(input_seq)
        target_seq = target_seq[:, :33]
        outputs = outputs.view(2, -1)
        target_seq = target_seq.float()
        outputs = outputs.float()
        # outputs = outputs.reshape(target_seq.shape[0], target_seq.shape[1])
        # print("Input: ",input_seq.shape)
        # print("Output: ",outputs.shape)
        # print("Target: ",target_seq.shape)
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
            outputs = model(input_seq)
            target_seq = target_seq[:, :33]
            outputs = outputs.view(2, -1)
            target_seq = target_seq.float()
            outputs = outputs.float()
            
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentences = get_sentences("temp_train.txt")
    # CHANGE THIS
    sentences = sentences[:100]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    word2vec_model = Word2Vec(sentences, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
    model = OutlineTransformer(EMBEDDING_DIM,word2vec_model.wv,NUM_HEADS,NUM_DECODERS,FF_DIM,DROPOUT_RATE,device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(num_chunks):
        train_data = dataloader_helper("temp_train.txt", "temp_train_target.txt", i * CHUNK_SIZE * BATCH_SIZE)
        # CHANGE THIS
        # train_data = train_data[:100]
        train_loader = get_data_loader(train_data, tokenizer,model,True)
        
        
        print(f"Training on chunk {i}")
        for epoch in range(10):
            train_loss = train(model, train_loader, optimizer, device,loss_fn)
            print(f"Epoch {epoch} train loss: {train_loss}")
            print(f"Epoch {epoch} eval loss: {evaluate(model, train_loader, device,loss_fn)}")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
