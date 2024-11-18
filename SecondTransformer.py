import re
import torch
import stanza
from DiscourseAwareStoryGen import train_step, DiscourseAwareStoryGenerator
from Outline import generate_outline, generate_abstract
from Evaluators import calculate_distinct_1_2, evaluate_coref_coherence, coherence_score, calculate_perplexity
from params import DISCOURSE_MARKERS, LAMBDA1, LAMBDA2, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EMBEDDING_DIM, NUM_CONNECTORS
from CorefClusters import coreference_loss
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, AutoTokenizer, GPT2Config, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

MAX_LEN = 200

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
            outline = generate_abstract(story)
            prompt, story, outline = preprocess(prompt), preprocess(story), preprocess(outline)
            input_dict = {'prompt': prompt, 'outline': outline, 'story': story}
            datalist.append(input_dict)
    return datalist

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
            
        return input_sequence, story

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

def get_bert_loss(model, outputs, golden_bert, golden_bert_tokenizer, discourse_model, tokenizer):
    output_indices = torch.argmax(outputs, dim=-1) 
    decoded_stories = []
    for sequence in output_indices:
        decoded_story = tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_stories.append(decoded_story)
    loss_array = []
    for _, story in enumerate(decoded_stories):
        loss_val = get_average_loss(story, discourse_model, golden_bert, golden_bert_tokenizer)
        if loss_val != -1:
            loss_array.append(loss_val)
    if len(loss_array) == 0:
        return -1
    return sum(loss_array) / len(loss_array)

def decode_output_list(model, outputs):
    output_indices = torch.argmax(outputs, dim=-1) 
    decoded_sentences = []
    for sequence in output_indices:
        decoded_sentence = model.tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    return decoded_sentences

def get_corenlp_loss(model, outputs, batch_attention_weights_list, nlp, tokenizer):
    output_indices = torch.argmax(outputs, dim=-1)
    decoded_stories = []
    for sequence in output_indices:
        decoded_story = tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_stories.append(decoded_story)
    loss_array = []
    for i, story in enumerate(decoded_stories):
        attention_weights = batch_attention_weights_list[i]
        loss_val = float(coreference_loss(attention_weights, story, nlp))
        loss_array.append(loss_val)
    return sum(loss_array) / len(loss_array) if loss_array else 0.0

def train(model, train_loader, optimizer, device, loss_function, golden_bert, golden_bert_tokenizer, discourse_model, nlp, tokenizer):
    model.train()
    total_loss = 0
    for input_seq, target_seq in train_loader:
        input_texts = [seq for seq in input_seq]
        target_texts = [seq for seq in target_seq]

        input_ids = tokenizer(input_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)
        target_ids = tokenizer(target_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=target_ids)
        output_ids = outputs.logits.argmax(dim=-1)
        lst  = []
        for output_id in output_ids[0]:
            output_text = tokenizer.decode([output_id.item()], skip_special_tokens=True)
            lst.append(output_text)
        # print("Output:", ' '.join(lst))
        generation_loss = outputs.loss
        batch_attention_weights_list = outputs.attentions if outputs.attentions is not None else None

        bert_loss = get_bert_loss(model, outputs.logits, golden_bert, golden_bert_tokenizer, discourse_model, tokenizer)
        try:
            corenlp_float_loss = get_corenlp_loss(model, outputs.logits, batch_attention_weights_list, nlp, tokenizer)
        except Exception:
            corenlp_float_loss = 0.0
        corenlp_loss = torch.tensor(corenlp_float_loss, device=device, requires_grad=True)
        
        total_loss_for_batch = generation_loss
        if bert_loss != -1:
            total_loss_for_batch += LAMBDA1 * bert_loss
        total_loss_for_batch += LAMBDA2 * corenlp_loss
        
        total_loss_for_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_for_batch.item()

    return total_loss / len(train_loader)

def decode_batch_outputs(tokenizer, outputs):
    output_indices = torch.argmax(outputs, dim=-1)
    decoded_texts = []
    for sequence in output_indices:
        decoded_text = tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_texts.append(decoded_text)
    return decoded_texts
    
def evaluate(model, loader, device, golden_bert, golden_bert_tokenizer, nlp, tokenizer):
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
            input_texts = [seq for seq in input_seq]
            target_texts = [seq for seq in target_seq]
            
            input_ids = tokenizer(input_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)
            target_ids = tokenizer(target_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)
            outputs = model(input_ids=input_ids, labels=target_ids)

            loss = outputs.loss
            total_loss += loss
            lst = decode_batch_outputs(tokenizer,outputs.logits)
            
            perplexity = calculate_perplexity(loss)
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
    
    return total_loss / len(loader)

def model_initializer(device):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    config = GPT2Config(n_layer=12)
    model = GPT2LMHeadModel(config).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    return model, tokenizer, optimizer, loss_fn

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

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder = BertModel.from_pretrained('bert-base-uncased')

    golden_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(DISCOURSE_MARKERS)).to(device)
    state_dict = torch.load("golden_bert.pt")
    golden_bert.load_state_dict(state_dict)  # Load weights
    golden_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    discourse_model = DiscourseAwareStoryGenerator(encoder=encoder, hidden_size=EMBEDDING_DIM, output_size=NUM_CONNECTORS, tokenizer=bert_tokenizer, device=device).to(device)

    print("Loaded Models")

    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,coref')

    import time
    for epoch in range(NUM_EPOCHS):
        st = time.time()
        train_loss = train(model, train_loader, optimizer, device, loss_fn, golden_bert, golden_bert_tokenizer, discourse_model, nlp, tokenizer)
        print(f"Epoch {epoch+1} train loss: {train_loss}")
        print("Evaluate Loss", evaluate(model, val_loader, device, golden_bert, golden_bert_tokenizer, nlp, tokenizer))
        print(time.time() - st)
    print("Test Loss", evaluate(model, test_loader, device, golden_bert, golden_bert_tokenizer, nlp, tokenizer))
    torch.save(model.state_dict(), "T2.pth")

if __name__ == "__main__":
    main()
