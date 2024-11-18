import torch
from params import BATCH_SIZE, MAX_LEN, LEARNING_RATE, NUM_EPOCHS, DISCOURSE_MARKERS
from torch.utils.data import Dataset, DataLoader
from Transformer import Transformer
from Evaluators import calculate_distinct_1_2, calculate_perplexity, evaluate_coref_coherence, coherence_score
from Outline import generate_outline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,  AutoTokenizer
from transformers import BertForSequenceClassification, BertTokenizer
import re
import stanza

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
    dataset = TextDataset(tokenized_data, tokenizer, train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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

def process_stories(stories, golden_bert, golden_bert_tokenizer, nlp, device):
    total_distinct_1 = 0
    total_distinct_2 = 0
    total_coref = 0
    total_bert = 0
    
    num_stories = len(stories)
    if num_stories == 0:
        return 0, 0, 0, 0
    
    for story in stories:
        distinct_1, distinct_2 = calculate_distinct_1_2(story)
        coref_coherence = evaluate_coref_coherence(story, nlp)
        bert_coher = coherence_score(story, golden_bert, golden_bert_tokenizer, device)
        
        total_distinct_1 += distinct_1
        total_distinct_2 += distinct_2
        total_coref += coref_coherence
        total_bert += bert_coher
    
    return (
        total_distinct_1 / num_stories,
        total_distinct_2 / num_stories,
        total_coref / num_stories,
        total_bert / num_stories
    )
    
def evaluate(model, loader, device, loss_function,golden_bert, golden_bert_tokenizer, nlp):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_distinct_1 = 0
    total_distinct_2 = 0
    total_coref = 0
    total_bert = 0
    num_batches = 0
    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs, batch_attention_weights_list = model(input_seq, None)
            lst = decode_output_list(model,outputs)
            # decode_output(model,outputs)
            outputs = outputs.view(-1, outputs.size(-1)) 
            target_seq = target_seq.view(-1)              
            loss = loss_function(outputs, target_seq)
            total_loss += loss.item()
            perplexity = calculate_perplexity(loss.item())
            distinct_1, distinct_2, coref_coherence, bert_coher = process_stories(lst, golden_bert, golden_bert_tokenizer, nlp, device)
            
            total_perplexity += perplexity
            total_distinct_1 += distinct_1
            total_distinct_2 += distinct_2
            total_coref += coref_coherence
            total_bert += bert_coher
            num_batches += 1
            
    print("\nOverall Averages:")
    print(f"Avg Perplexity: {total_perplexity/num_batches:.3f}")
    print(f"Avg Distinct-1: {total_distinct_1/num_batches:.3f}")
    print(f"Avg Distinct-2: {total_distinct_2/num_batches:.3f}")
    print(f"Avg Coref coherence: {total_coref/num_batches:.3f}")
    print(f"Avg BERT coherence: {total_bert/num_batches:.3f}")
    
    return total_loss / len(loader), distinct_1 / len(loader), distinct_2 / len(loader)

def model_initializer(device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = Transformer(tokenizer=tokenizer, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # as per the paper
    loss_fn = torch.nn.CrossEntropyLoss()
    return model, tokenizer, optimizer, loss_fn

def main():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model, tokenizer, optimizer, loss_fn = model_initializer(device)
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,coref')
    golden_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(DISCOURSE_MARKERS)).to(device)
    golden_bert.load_state_dict(torch.load('golden_bert.pt'))
    golden_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Loaded Models")

    # replace train16.wp_source and train16.wp_target with the actual file names
    train_data = dataloader_helper("train16.wp_source", "train16.wp_target")
    train_loader = get_data_loader(train_data, tokenizer, model, True)
    val_data = dataloader_helper("train16.wp_source", "train16.wp_target")
    val_loader = get_data_loader(val_data, tokenizer, model, False)
    test_data = dataloader_helper("train16.wp_source", "train16.wp_target")
    test_loader = get_data_loader(test_data, tokenizer, model, False)
    print("Loaded Dataloader")
    
    import time
    for epoch in range(NUM_EPOCHS):
        st = time.time()    
        train_loss = train(model, train_loader, optimizer, device, loss_fn)
        print(evaluate(model, val_loader, device, loss_fn,golden_bert, golden_bert_tokenizer, nlp)) # CHANGE THIS TO VALIDATION_LOADER
        print(f"Epoch {epoch+1} train loss: {train_loss}")
        print(time.time() - st)
    print(evaluate(model, test_loader, device, loss_fn,golden_bert, golden_bert_tokenizer, nlp))
    torch.save(model.state_dict(), 'T2.pt')

if __name__ == "__main__":
    main()