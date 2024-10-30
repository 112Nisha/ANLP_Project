import re
import torch
from torch.nn.functional import softmax
from stanfordnlp.server import CoreNLPClient
from transformers import BertTokenizer, BertForSequenceClassification

BATCH_SIZE = 8

def calculate_perplexity(loss_value):
    return torch.exp(torch.tensor(loss_value))

# Assuming there's only one generated text
# Pass words
def calculate_distinct_1_2(generated_text):
    total_1_grams = []
    total_2_grams = []
    
    words = generated_text.split()
    total_1_grams.extend(words)
    for index in range(len(words) - 1):
        total_2_grams.append((words[index], words[index + 1]))
    unique_1_grams = set(total_1_grams)
    unique_2_grams = set(total_2_grams)

    # I'm adding a small term to avoid division by 0, can we just set distinct 1/2 to 0 if len is 0??
    distinct_1 = len(unique_1_grams) / (len(total_1_grams) + 1e-10) 
    distinct_2 = len(unique_2_grams) / (len(total_2_grams) + 1e-10)  
    
    return distinct_1, distinct_2

# Pass words
# Run this command - java -mx8g -cp "/home/dell/github/ANLP_Project/stanford-corenlp-4.5.7/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 60000
# Make sure java is installed
# Push stanford CoreNLP object to gitrepo
# pip install stanfordnlp
# constraint on number of letters, goes to 500 as of now, might have to consider individual sentences and not
# the entire output batch
def calculate_coreference_chains(generated_text):
    with CoreNLPClient(timeout=60000, verbose=True) as client:
        annotations = client.annotate(generated_text, output_format='json', annotators='tokenize,ssplit,pos,lemma,ner,parse,coref')
        corefs = annotations['corefs']
        chains = {}
        for coref_id, coref_info in corefs.items():
            if isinstance(coref_info, list):
                chains[coref_id] = [mention['text'] for mention in coref_info]
        
        coherence_score = average_coreference_chain_length(chains)
        return coherence_score

def average_coreference_chain_length(coreference_chains):
    total_length = 0
    total_chains = len(coreference_chains)
    for chain in coreference_chains.values():
        total_length += len(chain)
        if total_chains > 0:
            average_length = total_length / total_chains
        else:
            average_length = 0
    return average_length

# fine tune
def fine_tune_bert():
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=9)

    return model, tokenizer

def tag_sentence_pairs(predicted_output, target_output, model, tokenizer, threshold=0.1):
    tagged_pairs = []
    for pred, targ in zip(predicted_output, target_output):
        inputs = tokenizer(pred, targ, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted_label = torch.max(probabilities, dim=1)
            if confidence >= threshold:
                label = predicted_label.item()
            else:
                # 8 is for unknown
                label = 8  

            tagged_pairs.append({
                "predicted": pred,
                "target": targ,
                "label": label,
                "confidence": confidence.item()
            })
    return tagged_pairs

# 8 is for unknown
def compute_unknown_percentage(tagged_pairs):
    unknown_count = sum(1 for pair in tagged_pairs if pair['label'] == 8)  
    total_count = len(tagged_pairs)

    if total_count == 0:
        return 0

    percentage = (unknown_count / total_count) * 100
    return percentage

def get_nth_line_from_file(filename,line_number):
    curr_index = 0
    with open(filename, 'r') as file:
        for line in file:
            if curr_index == line_number:
                return line.strip()
            curr_index += 1
    return None

def preprocess(text):
    cleaned_text = text.replace("<newline>", "")
    output = re.sub(r'([.,!?;:*()"\'“”‘’_\u2014-])', r' \1 ', cleaned_text)
    output = re.sub(r'[^\w\s]', '', output)
    output = re.sub(r'\s+', ' ', output)
    output = output.strip()
    output = output.lower()
    return output

def dataloader_helper(source_filename, target_filename,start_index):
    datalist = []
    for curr_index in range(BATCH_SIZE):
        prompt = get_nth_line_from_file(source_filename,start_index+curr_index)
        story = get_nth_line_from_file(target_filename,start_index+curr_index)
        # CHANGE THIS LATER
        outline = prompt
        prompt = preprocess(prompt)
        story = preprocess(story)
        outline = preprocess(outline)
        input_dict = {'prompt':prompt,'outline':outline,'story':story}
        datalist.append(input_dict)
    return datalist

def main():
    input = ["today was a nice day", "i liked it"]
    output = ["today was a nice day","i liked it"]
    model, tokenizer = fine_tune_bert()
    tagged_pairs = tag_sentence_pairs(output,input,model,tokenizer,threshold=0.2)
    print(tagged_pairs)
    val = compute_unknown_percentage(tagged_pairs)
    print(val)
    pass

if __name__ == "__main__":
    main()
