import re
import torch
from transformers import GPT2Tokenizer
from utils import get_nth_line_from_file
from torch.utils.data import Dataset, DataLoader
from story_transformer import StoryTransformer
from params import BATCH_SIZE, NUM_EPOCHS,CHUNK_SIZE, MAX_LEN, LEARNING_RATE

# SHOULD WE USE DROPOUT AND A SCHEDULER TO REDUCE OVERFITTING?
# SHOULD WE BE USING THE GPT2 TOKENIZER? bECAUSE IT'S FOR SUMMARIZATION TASKS SPECIFICALLY.
# Consider using a learning rate scheduler
# Add gradient clipping to prevent exploding gradients

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

        input_sequence = prompt + " <s> " + outline + " <sep> "
        if self.isTrain:
            input_sequence += story

        
        input_sequence_indices = self.tokenizer.encode(input_sequence, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').squeeze()
        if story:
            story_indices = self.tokenizer.encode(story, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').squeeze()
        else:
            pad_index = self.word2index['<pad>']
            story_indices = torch.full((MAX_LEN,), pad_index, dtype=torch.long)
        return input_sequence_indices, story_indices


def get_data_loader(tokenized_data, tokenizer, model, train=True):
    dataset = TextDataset(tokenized_data, tokenizer, model.word_to_index, model.index_to_word, train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def decode_output(model, outputs):
    output_indices = torch.argmax(outputs, dim=-1) 

    decoded_sentences = []
    for sequence in output_indices:
        decoded_sentence = model.tokenizer.decode(sequence, skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)

    for i, sentence in enumerate(decoded_sentences):
        print(f"Decoded Sentence {i+1}: ", sentence)
        print("\n")


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
        # [batch_size, sequence_length, vocab_size]
        outputs = model(input_seq, target_seq) 
        # decode_output(model,outputs)
        # Reshape to [batch_size * sequence_length, vocab_size]
        outputs = outputs.view(-1, outputs.size(-1))
        target_seq = target_seq.view(-1)             

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
            outputs = outputs.view(-1, outputs.size(-1))     
            target_seq = target_seq.view(-1)              
            loss = loss_function(outputs, target_seq)
            total_loss += loss.item()
    return total_loss / len(loader)

def model_initializer(sentences,device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = StoryTransformer(tokenizer=tokenizer, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.word_to_index['<pad>'])
    return model, tokenizer, optimizer, loss_fn

def main():
    # CHANGE FILE NAMES
    with open("temp_train.txt", 'r') as fp:
        lines = len(fp.readlines())
    num_loops = (lines//(BATCH_SIZE*CHUNK_SIZE))+1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentences_train = get_sentences("temp_train.txt")
    sentences_target = get_sentences("temp_train_target.txt")
    sentences = sentences_train + sentences_target
    model, tokenizer, optimizer, loss_fn = model_initializer(sentences,device)
    for i in range(num_loops):
        train_data = dataloader_helper("temp_train.txt", "temp_train_target.txt", i * CHUNK_SIZE * BATCH_SIZE)
        train_loader = get_data_loader(train_data, tokenizer, model, True)
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
