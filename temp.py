import torch
import re
from stanfordnlp.server import CoreNLPClient

def calculate_perplexity(loss_value):
    loss_tensor = torch.tensor(loss_value)
    perplexity = torch.exp(loss_tensor)
    return perplexity

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
def calculate_coreference_chains(generated_text):
    with CoreNLPClient(timeout=30000, verbose=True) as client:
        annotations = client.annotate(generated_text, output_format='json', annotators='tokenize,ssplit,pos,lemma,ner,parse,coref')
        corefs = annotations['corefs']
        chains = {}
        for coref_id, coref_info in corefs.items():
            chains[coref_id] = [mention['text'] for mention in coref_info]

        return chains

def get_nth_line_from_file(filename,line_number):
    curr_index = 0
    with open(filename, 'r') as file:
        for line in file:
            if curr_index == line_number:
                return line.strip()
            curr_index += 1
    return None

BATCH_SIZE = 8

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
    sentence = "B likes apples. She eats them all the time."
    chains = calculate_coreference_chains(sentence)
    print(chains)
    pass

if _name_ == "_main_":
    main()