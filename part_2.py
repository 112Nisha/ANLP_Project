import re
import torch
from story_transformer import StoryTransformer
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, BertTokenizer, BertModel
from golden_BERT import model_initializer as model_initializer_bert
import stanza
from get_coreference_consistency import get_coreference_clusters
stanza.download('en')
from discourse_aware_story_gen import train_step, DiscourseAwareStoryGenerator
from params import BATCH_SIZE, CHUNK_SIZE, MAX_LEN, LEARNING_RATE, NUM_CONNECTORS, EMBEDDING_DIM, NUM_EPOCHS, LAMBDA1, LAMBDA2

tokenizer = GPT2Tokenizer

def get_nth_line_from_file(file, n):
    with open(file, 'r') as file:
        for current_line_number, line in enumerate(file, start=0):
            if current_line_number == n:
                return line.strip()
    return None

def get_average_loss(generated_text, model, golden_bert, golden_bert_tokenizer):
    sentences = re.split(r'(?<=[.!?]) +', generated_text.strip())
    sentence_pairs = []
    for i in range(0, len(sentences)-1):
        pair = sentences[i:i+2]
        sentence_pairs.append(pair)
    loss_arr = []
    for i, pair in enumerate(sentence_pairs):
        loss_val = train_step(model=model, sentences=pair, golden_bert=golden_bert, golden_bert_tokenizer=golden_bert_tokenizer)
        loss_arr.append(loss_val)
    if len(loss_arr) == 0:
        return -1
    return sum(loss_arr) / len(loss_arr)

def preprocess(text):
    if text is None:
        return ""
    cleaned_text = text.replace("<newline>", "")
    # output = re.sub(r'[^\w\s.]', '', cleaned_text)
    # output = re.sub(r'\s+', ' ', output) 
    # output = output.strip().lower()
    # return output
    return cleaned_text

def dataloader_helper(source_filename, target_filename, start_index):
    datalist = []
    for curr_index in range(CHUNK_SIZE * BATCH_SIZE):
        prompt, story = get_nth_line_from_file(source_filename, start_index + curr_index), get_nth_line_from_file(target_filename, start_index + curr_index)
        # print(prompt)
        outline = prompt # CHANGE THIS LATER
        if not prompt: 
            continue
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
        decoded_sentence = model.tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    # For each chunk, it will print the decoded sentences per batch so if chunk_size = 6 and batch_size = 2
    # there will be 3 such runs of this loop
    for i, sentence in enumerate(decoded_sentences):
        print(f"Decoded Sentence {i+1}: \n", sentence)

def get_bert_loss(model, outputs, golden_bert, golden_bert_tokenizer, discourse_model):
    output_indices = torch.argmax(outputs, dim=-1) 

    decoded_stories = []
    for sequence in output_indices:
        decoded_story = model.tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_stories.append(decoded_story)

    loss_array = []
    for _, story in enumerate(decoded_stories):
        loss_val = get_average_loss(story, discourse_model, golden_bert, golden_bert_tokenizer)
        # print(f"\033[92m{type(loss_val), loss_val}\033[00m")
        if loss_val != -1:
            loss_array.append(loss_val)
    if len(loss_array) == 0:
        return -1
    return sum(loss_array) / len(loss_array)

def get_strings(logits):
    token_ids = torch.argmax(logits, dim=-1)  # Shape: [batch_size, sequence_length]
    
    # Convert token IDs to words using the tokenizer
    generated_strings = []
    for tokens in token_ids:
        words = tokenizer.convert_ids_to_tokens(tokens.tolist(), skip_special_tokens=True)
        generated_strings.append(tokenizer.convert_tokens_to_string(words))
    
    return generated_strings

def get_coreference_loss(attention_weights, coreference_clusters, input_sequence):
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    attention_weights = attention_weights.mean(dim=1)  # Average over heads, shape: (batch_size, seq_len, seq_len)

    coref_loss = 0
    total_mentions = 0
    
    # Loop through each batch
    for batch_idx in range(batch_size):
        sentence_indices = input_sequence[batch_idx]  # Shape: (seq_len,)
        sentence_tokens = [GPT2Tokenizer.decode([idx.item()]) for idx in sentence_indices]

        for cluster in coreference_clusters[batch_idx]:
            cluster_word = cluster["text"]
            cluster_token_indices = [
                idx for idx, token in enumerate(sentence_tokens) if token == cluster_word
            ]
            
            # Ensure indices are from the correct dictionary
            cluster_token_indices = torch.tensor(cluster_token_indices)
            
            # Calculate the attention sum for tokens in the cluster
            for i in cluster_token_indices:
                # Sum of log attention weights for tokens in the same cluster
                attention_sum = attention_weights[batch_idx, i, cluster_token_indices].sum()
                log_attention_sum = torch.log(attention_sum + 1e-8)
                coref_loss -= log_attention_sum
                total_mentions += 1
    
    # Normalize the coref loss
    if total_mentions > 0:
        coref_loss /= total_mentions
    
    return coref_loss
 
def train(model, train_loader, optimizer, device, loss_function, golden_bert, golden_bert_tokenizer, discourse_model):
    model.train()
    total_loss = 0
    for input_seq, target_seq in train_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        outputs, attention_weights = model(input_seq, target_seq, get_attention_weights=True)       # [batch_size, sequence_length, vocab_size]
        bert_loss = get_bert_loss(model, outputs, golden_bert, golden_bert_tokenizer, discourse_model)
        print(f"BERT Loss: {bert_loss}")
        coreference_clusters = get_coreference_clusters(get_strings(outputs))
        get_coreference_loss = get_coreference_loss(attention_weights, coreference_clusters, target_seq)
        outputs = outputs.view(-1, outputs.size(-1)) # Reshape to [batch_size * sequence_length, vocab_size]
        target_seq = target_seq.view(-1)             # Reshape to [batch_size * sequence_length]

        loss = loss_function(outputs, target_seq)
        bert_loss = get_bert_loss(model, outputs, golden_bert, golden_bert_tokenizer, discourse_model)
        if bert_loss != -1:
            loss.backward(retain_graph=True)  # Retain the graph for subsequent backward passes
            scaled_bert_loss = LAMBDA1 * bert_loss
            scaled_bert_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        total_loss += loss.item() # add BERT loss
    return total_loss / len(train_loader)

# Do I add bert loss here too?
def evaluate(model, loader, device, loss_function):
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

def model_initializer(device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = StoryTransformer(tokenizer=tokenizer, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # as per the paper
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.word_to_index['<pad>']) # CHANGE THIS TO ACCOUNT FOR LOSSES FROM BERT AND CORENLP
    return model, tokenizer, optimizer, loss_fn

def main():
    # CHANGE FILE NAMES
    with open("temp_train.txt", 'r') as fp:
        lines = len(fp.readlines())
    num_loops = (lines // (BATCH_SIZE * CHUNK_SIZE)) + 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, optimizer, loss_fn = model_initializer(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder = BertModel.from_pretrained('bert-base-uncased')

    golden_bert, _ , _, _ = model_initializer_bert(device) # CHANGE THIS 
    # golden_bert = torch.load("bert_model.pth")

    discourse_model = DiscourseAwareStoryGenerator(encoder=encoder, hidden_size=EMBEDDING_DIM, output_size=NUM_CONNECTORS,tokenizer=tokenizer, device=device)
    for i in range(num_loops):
        train_data = dataloader_helper("temp_train.txt", "temp_train_target.txt", i * CHUNK_SIZE * BATCH_SIZE)
        train_loader = get_data_loader(train_data, tokenizer, model, True)
        print(f"Training on chunk {i+1}")
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, train_loader, optimizer, device, loss_fn, golden_bert, tokenizer, discourse_model)
            eval_loss = evaluate(model, train_loader, device, loss_fn) # CHANGE THIS TO VALIDATION_LOADER
            print(f"Epoch {epoch+1} train loss: {train_loss}")
            print(f"Epoch {epoch+1} eval loss: {eval_loss}")

    torch.save(model.state_dict(), "transformer_2.pth")
    # torch.save(discourse_model.state_dict(), "discourse_model.pth")

if __name__ == "__main__":
    main()
