import torch
from Outline import generate_outline, generate_abstract
from Evaluators import calculate_perplexity, calculate_distinct_1_2
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config,  AutoTokenizer
from params import BATCH_SIZE, MAX_LEN, LEARNING_RATE, NUM_EPOCHS
import re

def get_nth_line_from_file(file, n):
    with open(file, 'r') as file:
        for current_line_number, line in enumerate(file, start=0):
            if current_line_number == n:
                return line.strip()
    return None

def preprocess(text):
    if text is None:
        return ""
    cleaned_text = text.replace("<newline>", "")
    cleaned_text = cleaned_text.replace('\n', '')
    cleaned_text = cleaned_text.strip()  
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    output = re.sub(r'[^\w\s.]', '', cleaned_text)
    output = output.strip().lower()
    return output

def dataloader_helper(source_filename, target_filename):
    datalist = []
    with open(source_filename, 'r') as source_file, open(target_filename, 'r') as target_file:
        source_lines = source_file.readlines()
        target_lines = target_file.readlines()
        for prompt, story in zip(source_lines, target_lines):
            if not prompt.strip():
                continue
            outline = generate_outline(story)
            # outline = generate_abstract(story)
            prompt, story, outline = preprocess(prompt), preprocess(story), preprocess(outline)
            input_dict = {'prompt': prompt, 'outline': outline, 'story': story}
            datalist.append(input_dict)
    return datalist

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, train=False):
        self.data = data
        self.isTrain = train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, outline = self.data[idx]['prompt'], self.data[idx]['outline']
        outline = ' '.join(outline).replace('\n', '')
        outline = preprocess(outline)
        input_sequence = prompt + "<sep>"
        if self.isTrain:
            input_sequence += outline
        return input_sequence, outline

def get_data_loader(tokenized_data, tokenizer, model, train=True):
    dataset = TextDataset(tokenized_data, tokenizer, train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    # return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def decode_output(model, outputs):
    output_indices = torch.argmax(outputs, dim=-1) 
    decoded_sentences = []
    for sequence in output_indices:
        decoded_sentence = model.tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    for i, sentence in enumerate(decoded_sentences):
        print(f"Decoded Sentence {i+1}: \n", sentence)

def decode_output_gpt2(tokenizer, outputs):
    output_indices = torch.argmax(outputs.logits, dim=-1)  # shape will be [batch_size, sequence_length]
    print(type(output_indices))
    decoded_sentences = []
    for sequence in output_indices:
        decoded_sentence = tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    for i, sentence in enumerate(decoded_sentences):
        print(f"Decoded Sentence {i+1}:\n{sentence}")
    return decoded_sentences
        
def decode_output_list(model, outputs):
    output_indices = torch.argmax(outputs, dim=-1) 
    decoded_sentences = []
    for sequence in output_indices:
        decoded_sentence = model.tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    return decoded_sentences

def train(model, train_loader, optimizer, device, loss_function, tokenizer):
    model.train()
    total_loss = 0
    for input_seq, target_seq in train_loader:
        input_texts = [seq for seq in input_seq]
        target_texts = [seq for seq in target_seq]

        input_ids = tokenizer(input_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)
        target_ids = tokenizer(target_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)

        outputs = model(input_ids=input_ids, labels=target_ids)

        loss = outputs.loss
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(train_loader)

def evaluate(model, loader, device, loss_function, tokenizer):
    model.eval()
    total_loss = 0
    distinct_1 = 0
    distinct_2 = 0
    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_texts = [seq for seq in input_seq]
            target_texts = [seq for seq in target_seq]
            
            input_ids = tokenizer(input_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)
            target_ids = tokenizer(target_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)
            outputs = model(input_ids=input_ids, labels=target_ids)
            output_ids = outputs.logits.argmax(dim=-1)
            lst  = []
            for i in range(BATCH_SIZE):
                temp_lst = []
                for output_id in output_ids[i]:
                    output_text = tokenizer.decode([output_id.item()], skip_special_tokens=True)
                    temp_lst.append(output_text)
                lst.append(' '.join(temp_lst))
            temp_distinct_1 = []
            temp_distinct_2 = []
            for sentence in lst:
                d1,d2 = calculate_distinct_1_2(sentence)
                temp_distinct_1.append(d1)
                temp_distinct_2.append(d2)
            distinct_1 += sum(temp_distinct_1)/len(temp_distinct_1)
            distinct_2 += sum(temp_distinct_2)/len(temp_distinct_2)             
            loss = outputs.loss
            total_loss += loss
    return total_loss / len(loader), distinct_1 / len(loader), distinct_2 / len(loader)

def model_initializer(device):
    config = GPT2Config(n_layer=12)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # as per the paper
    loss_fn = torch.nn.CrossEntropyLoss()
    return model, tokenizer, optimizer, loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, optimizer, loss_fn = model_initializer(device)
    # replace train16.wp_source and train16.wp_target with the actual files
    train_data = dataloader_helper("train16.wp_source", "train16.wp_target")
    train_loader = get_data_loader(train_data, tokenizer, model, True)
    val_data = dataloader_helper("train16.wp_source", "train16.wp_target")
    val_loader = get_data_loader(val_data, tokenizer, model, False)
    test_data = dataloader_helper("train16.wp_source", "train16.wp_target")
    test_loader = get_data_loader(test_data, tokenizer, model, False)
    print("Loaded Dataloaders")

    import time
    for epoch in range(NUM_EPOCHS):
        st = time.time()    
        train_loss = train(model, train_loader, optimizer, device, loss_fn, tokenizer)
        print(evaluate(model, val_loader, device, loss_fn, tokenizer))
        print(f"Epoch {epoch+1} train loss: {train_loss}")
        print(time.time() - st)
    print(evaluate(model, test_loader, device, loss_fn, tokenizer))
    torch.save("T1.pt")

if __name__ == "__main__":
    main()
