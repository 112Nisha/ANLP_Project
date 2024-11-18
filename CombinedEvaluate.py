import torch
from Outline import generate_outline
from params import BATCH_SIZE, LEARNING_RATE, MAX_LEN, DISCOURSE_MARKERS
from Evaluators import calculate_distinct_1_2, evaluate_coref_coherence, coherence_score, calculate_perplexity
from Transformer import Transformer
import stanza
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BertTokenizer, BertForSequenceClassification
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

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, stage='first'):
        self.data = data
        self.tokenizer = tokenizer
        self.stage = stage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]['prompt']
        outline = self.data[idx]['outline'] if 'outline' in self.data[idx] else None
        story = self.data[idx]['story'] if 'story' in self.data[idx] else None

        if self.stage == 'first':
            input_sequence = prompt + " <s> "
        else:
            input_sequence = prompt + " <s> " + outline + " <sep> "
        if self.stage == 'first':
            target = outline if outline else ''
        else:
            target = story if story else ''
        return input_sequence, target

def decode_output_gpt2(tokenizer, outputs):
    output_indices = torch.argmax(outputs.logits, dim=-1)  # shape will be [batch_size, sequence_length]
    decoded_sentences = []
    for sequence in output_indices:
        decoded_sentence = tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    return decoded_sentences

def evaluate(model1, model2, loader, tokenizer, device, loss_function, golden_bert, golden_bert_tokenizer, nlp):
    model1.eval()
    model2.eval()
    total_loss = 0
    
    total_perplexity = 0
    total_distinct_1 = 0
    total_distinct_2 = 0
    total_coref = 0
    total_bert = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(loader):
            input_texts = [seq for seq in input_seq]
            target_texts = [seq for seq in target_seq]
            
            input_ids = tokenizer(input_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)
            target_ids = tokenizer(target_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').input_ids.to(device)
            
            outline_outputs = model1(input_ids=input_ids, labels=target_ids)
            outlines = decode_output_gpt2(tokenizer, outline_outputs)

            prompts = [item['prompt'] for item in loader.dataset.data[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]]
            stories = [item['story'] for item in loader.dataset.data[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]]
            tokenized_data = unit_dataloader(stories, prompts, outlines)
            dataset = TextDataset(tokenized_data, tokenizer, stage='second')
            second_stage_inputs = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            input_s, target_s = next(iter(second_stage_inputs))
            input_texts = [seq for seq in input_seq]
            target_texts = [seq for seq in target_seq]
            input_ids = tokenizer(input_texts, truncation=True, padding='max_length', max_length=200, return_tensors='pt').input_ids.to(device)
            target_ids = tokenizer(target_texts, truncation=True, padding='max_length', max_length=200, return_tensors='pt').input_ids.to(device)

            story_outputs = model2(input_ids=input_ids, labels=target_ids)
            stories = decode_output_gpt2(tokenizer, story_outputs)

            total_loss += story_outputs.loss
            
            perplexity = calculate_perplexity(story_outputs.loss)
            distinct_1, distinct_2, coref_coherence, bert_coher = process_stories(stories, golden_bert, golden_bert_tokenizer, nlp, device)
            
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

def dataloader_helper(source_filename, target_filename):
    datalist = []
    with open(source_filename, 'r') as source_file, open(target_filename, 'r') as target_file:
        source_lines = source_file.readlines()
        target_lines = target_file.readlines()
        for prompt, story in zip(source_lines, target_lines):
            if not prompt.strip():
                continue
            outline = generate_outline(story)
            prompt, story, outline = preprocess(prompt), preprocess(story), preprocess(outline)
            input_dict = {'prompt': prompt, 'outline': outline, 'story': story}
            datalist.append(input_dict)
    return datalist

def unit_dataloader(stories, prompts, outlines):
    datalist = []
    for curr_index in range(BATCH_SIZE):
        input_dict = {'prompt': prompts[curr_index], 'outline': outlines[curr_index], 'story':stories[curr_index]}
        datalist.append(input_dict)
    return datalist

def decode_batch_outputs(tokenizer, outputs):
    output_indices = torch.argmax(outputs, dim=-1)
    decoded_texts = []
    for sequence in output_indices:
        decoded_text = tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_texts.append(decoded_text)
    return decoded_texts

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

def model_initializer(device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = Transformer(tokenizer=tokenizer, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # as per the paper
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.word_to_index['<pad>'])
    return model, tokenizer, optimizer, loss_fn

def get_data_loader(tokenized_data, tokenizer, model, stage='first'):
    dataset = TextDataset(tokenized_data, tokenizer, stage=stage)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def main():  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tokenizer, optimizer, loss_fn = model_initializer(device)
    config = GPT2Config(n_layer=12)
    model1 = GPT2LMHeadModel(config).to(device)
    model2 = GPT2LMHeadModel(config).to(device)
    model2.config.pad_token_id = tokenizer.pad_token_id
    model1.config.pad_token_id = tokenizer.pad_token_id
    model1.load_state_dict(torch.load("T1.pt"))
    model2.load_state_dict(torch.load("T2.pth"))

    golden_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(DISCOURSE_MARKERS)).to(device)
    state_dict = torch.load("golden_bert.pt")
    golden_bert.load_state_dict(state_dict)  # Load weights
    golden_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Loaded Models")

    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,coref')
    test_data = dataloader_helper("train16.wp_source", "train16.wp_target")
    test_loader = get_data_loader(test_data, tokenizer, model1, 'first')
    print("Loaded Dataloaders")
    eval_loss = evaluate(model1, model2, test_loader, tokenizer, device, loss_fn, golden_bert, golden_bert_tokenizer, nlp)
    print(f"Eval loss: {eval_loss}")

if __name__ == "__main__":
    main()
