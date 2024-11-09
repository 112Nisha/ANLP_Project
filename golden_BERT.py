import torch
import pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer

# Constants
MAX_LENGTH = 128
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 0.001

DISCOURSE_MARKERS = [
    'and', 'but', 'because', 'when', 'if', 'so', 'before', 'though'
]

class DiscourseDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load TSV file
        df = pd.read_csv(file_path, sep='\t', names=['sentence1', 'sentence2', 'label'])
        self.sentence_pairs = list(zip(df['sentence1'], df['sentence2']))
        self.labels = [DISCOURSE_MARKERS.index(label) for label in df['label']]

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sent1, sent2 = self.sentence_pairs[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sent1,
            sent2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            total_preds.extend(predictions.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(total_labels, total_preds)
    return total_loss / len(loader), accuracy

def model_initializer(device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(DISCOURSE_MARKERS))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    return model, tokenizer, optimizer, loss_fn

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
        
def predict_discourse_marker(model, tokenizer, sentence1, sentence2, device):
   # print(f"Model type: {type(model)}")
    """
    Predicts the discourse marker between two input sentences
    """
    model.eval()
    
    # Tokenize input sentences
    encoding = tokenizer.encode_plus(
        sentence1,
        sentence2,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1)
        
    predicted_marker = DISCOURSE_MARKERS[prediction.item()]
    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction.item()].item()
    
    return predicted_marker, confidence

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, optimizer, loss_fn = model_initializer(device)

    train_dataset = DiscourseDataset('/kaggle/input/small-test-tsv/temp.tsvc', tokenizer) # Change to train.tsv
    # val_dataset = DiscourseDataset('test.tsv', tokenizer) # Change to val.tsv
    # test_dataset = DiscourseDataset('test.tsv', tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}')
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f'Train Loss: {train_loss}')
        # val_loss, val_accuracy = evaluate(model, val_loader, device)
        # print(f'Val Loss: {val_loss}')
        # print(f'Val Accuracy: {val_accuracy}')
        torch.save(model.state_dict(), "/kaggle/working/golden_bert.pth")
    # torch.save(model.state_dict(), "/kaggle/working/golden_bert.pth")
    # test_loss, test_accuracy = evaluate(model, test_loader, device)
    # print(f'Test Loss: {test_loss}')
    # print(f'Test Accuracy: {test_accuracy}')
    
    # Example of using the prediction function
    # sent1 = "Her eyes flew up to his face."
    # sent2 = "Suddenly she realized why he looked so different."
    # marker, confidence = predict_discourse_marker(model, tokenizer, sent1, sent2, device)
    # print(f"Predicted discourse marker: {marker} (confidence: {confidence:.2f})")

if __name__ == '__main__':
    main()
