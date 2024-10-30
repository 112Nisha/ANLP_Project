import torch
from outline_transformer import OutlineTransformer
from torch.utils.data import Dataset, DataLoader
import re
from params import MAX_LEN, BATCH_SIZE, CHUNK_SIZE
from utils import get_nth_line_from_file
from transformers import GPT2Tokenizer

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
    def __init__(self, data, tokenizer, train=False):
        self.data = data
        self.isTrain = train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_sequence = self.data[idx]['prompt'].tolist() + "<s>" + self.data[idx]['outline'].tolist() + "<sep>"
        if self.train:
            input_sequence += self.data[idx]['story'].tolist()
        input_sequence = torch.tensor(self.tokenizer.encode(input_sequence))
        return input_sequence[:MAX_LEN], self.data[idx]['story']

def get_data_loader(tokenized_data, train=True):
    dataset = TextDataset(tokenized_data, train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for input_seq, target_seq in train_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        outputs = model(input_seq)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    with open("writingPrompts/train.wp_source", 'r') as fp:
        lines = len(fp.readlines())
    num_chunks = lines // CHUNK_SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OutlineTransformer(device=device, embedding_matrix=None)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    for i in range(num_chunks):
        train_data = dataloader_helper("writingPrompts/train.wp_source", "writingPrompts/train.wp_target", i * CHUNK_SIZE * BATCH_SIZE)
        train_loader = get_data_loader(train_data, True)
        print(f"Training on chunk {i}")
        for epoch in range(10):
            train_loss = train(model, train_loader, optimizer, device)
            print(f"Epoch {epoch} train loss: {train_loss}")
            print(f"Epoch {epoch} eval loss: {evaluate(model, train_loader, device)}")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
