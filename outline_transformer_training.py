import torch
from Transformer import Transformer
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from get_outline import generate_outline
from params import BATCH_SIZE, MAX_LEN, LEARNING_RATE, NUM_EPOCHS

def get_nth_line_from_file(file, n):
    with open(file, 'r') as file:
        for current_line_number, line in enumerate(file, start=0):
            if current_line_number == n:
                return line.strip()
    return None

def preprocess(text):
    if text is None:
        return ""
    # print the text in green
    cleaned_text = text.replace("<newline>", "")
    # output = re.sub(r'[^\w\s.]', '', cleaned_text)
    # output = re.sub(r'\s+', ' ', output) 
    # output = output.strip().lower()
    # return output
    return cleaned_text

def dataloader_helper(source_filename, target_filename, start_index, num_lines):
    datalist = []
    for curr_index in range(num_lines):
        prompt, story = get_nth_line_from_file(source_filename, start_index + curr_index), get_nth_line_from_file(target_filename, start_index + curr_index)
        if not prompt:
            continue
        prompt, story = preprocess(prompt), preprocess(story)
        outline = generate_outline(story)
        input_dict = {'prompt': prompt, 'outline': outline}
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
        prompt, outline = self.data[idx]['prompt'], self.data[idx]['outline']
        outline = ' '.join(outline).replace('\n', ' ')
        input_sequence = prompt + " <sep> "
        if self.isTrain:
            input_sequence += ''.join(outline)

        input_sequence_indices = self.tokenizer.encode(input_sequence, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').squeeze()
        if outline:
            outline_indices = self.tokenizer.encode(outline, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').squeeze()
        else:
            pad_index = self.word2index['<pad>']
            outline_indices = torch.full((MAX_LEN,), pad_index, dtype=torch.long)
        return input_sequence_indices, outline_indices

def get_data_loader(tokenized_data, tokenizer, model, train=True):
    dataset = TextDataset(tokenized_data, tokenizer, model.word_to_index, model.index_to_word, train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def decode_output(model, outputs):
    output_indices = torch.argmax(outputs, dim=-1) 
    decoded_sentences = []
    for sequence in output_indices:
        decoded_sentence = model.tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    # For each chunk, it will print the decoded sentences per batch so if chunk_size = 6 and batch_size = 2
    # there will be 3 such runs of this loop
    for i, sentence in enumerate(decoded_sentences):
        print(f"Decoded Sentence {i+1}: \n", sentence)

def train(model, train_loader, optimizer, device, loss_function):
    model.train()
    total_loss = 0
    for input_seq, target_seq in train_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        outputs, batch_attention_weights_list = model(input_seq, target_seq)       # [batch_size, sequence_length, vocab_size]
        outputs = outputs.view(-1, outputs.size(-1)) # Reshape to [batch_size * sequence_length, vocab_size]
        target_seq = target_seq.view(-1)             # Reshape to [batch_size * sequence_length]
        loss = loss_function(outputs, target_seq)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() # add BERT loss
    return total_loss / len(train_loader)

def evaluate(model, loader, device, loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs, batch_attention_weights_list = model(input_seq, target_seq)
            # decode_output(model,outputs)
            outputs = outputs.view(-1, outputs.size(-1)) 
            target_seq = target_seq.view(-1)              
            loss = loss_function(outputs, target_seq)
            total_loss += loss.item()
    return total_loss / len(loader)

def model_initializer(device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = Transformer(tokenizer=tokenizer, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # as per the paper
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.word_to_index['<pad>'])
    return model, tokenizer, optimizer, loss_fn

def main():
    # CHANGE FILE NAMES
    with open("temp_train.txt", 'r') as fp:
        lines = len(fp.readlines())
    # num_loops = (lines // (BATCH_SIZE * CHUNK_SIZE)) + 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, optimizer, loss_fn = model_initializer(device)

    train_data = dataloader_helper("temp_train.txt", "temp_train_target.txt", 0, lines)
    train_loader = get_data_loader(train_data, tokenizer, model, True)
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, device, loss_fn)
        eval_loss = evaluate(model, train_loader, device, loss_fn) # CHANGE THIS TO VALIDATION_LOADER
        print(f"Epoch {epoch+1} train loss: {train_loss}")
        print(f"Epoch {epoch+1} eval loss: {eval_loss}")

    torch.save(model.state_dict(), "transformer_1.pth")

if __name__ == "__main__":
    main()
