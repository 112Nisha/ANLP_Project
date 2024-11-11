import torch
from params import THRESHOLD
from transformers import BertTokenizer
# from part_2 import dataloader_helper
from discourse_aware_story_gen import predict_discourse_marker
import stanza
import logging
logging.getLogger("stanza").setLevel(logging.ERROR) 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

def average_mentions_per_coref_chain(story,nlp):
    doc = nlp(story)
    coref_chains = doc.coref
    if not coref_chains:
        return 0
    total_mentions = sum(len(chain.mentions) for chain in coref_chains)
    avg_mentions = total_mentions / len(coref_chains)
    return avg_mentions

# Pass words
# Run this command - java -mx8g -cp "/home/dell/github/ANLP_Project/stanford-corenlp-4.5.7/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 60000
# Make sure java is installed
# Push stanford CoreNLP object to gitrepo
# pip install stanfordnlp
# constraint on number of letters, goes to 500 as of now, might have to consider individual sentences and not
# the entire output batch
# def calculate_coreference_chains(generated_text):
#     with CoreNLPClient(timeout=60000, verbose=True) as client:
#         annotations = client.annotate(generated_text, output_format='json', annotators='tokenize,ssplit,pos,lemma,ner,parse,coref')
#         corefs = annotations['corefs']
#         chains = {}
#         for coref_id, coref_info in corefs.items():
#             if isinstance(coref_info, list):
#                 chains[coref_id] = [mention['text'] for mention in coref_info]
        
#         coherence_score = average_coreference_chain_length(chains)
#         return coherence_score

# def average_coreference_chain_length(coreference_chains):
#     total_length = 0
#     total_chains = len(coreference_chains)
#     for chain in coreference_chains.values():
#         total_length += len(chain)
#         if total_chains > 0:
#             average_length = total_length / total_chains
#         else:
#             average_length = 0
#     return average_length

def coherence_score(generated_text):
    golden_bert = torch.load("bert_model.pth")
    golden_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generated_text = generated_text.split(".")
    num_unknown = 0
    total = len(generated_text)

    for i in range(total-1):
        _, confidence = predict_discourse_marker(golden_bert, golden_bert_tokenizer, generated_text[i], generated_text[i+1], golden_bert.device)
        if confidence < THRESHOLD:
            num_unknown += 1
    return num_unknown / total

def main():
    story = '''
There was once a pleased elephant who generally harassed smaller animals.
He would go to the ant colony and shower water on the ants.
The ants, with their size, could just cry.
The elephant laughed and threatened the ants that he would kill them.
The ants had enough and chose to show the elephant a lesson.
They went straight into the elephant's trunk and started messing with him.
The elephant started crying in pain. He understood his mistake and apologised to the ants
and every one of the animals he had harassed.
There was once a pleased elephant who generally harassed smaller animals.
He would go to the ant colony and shower water on the ants.
The ants, with their size, could just cry.
The elephant laughed and threatened the ants that he would kill them.
The ants had enough and chose to show the elephant a lesson.
They went straight into the elephant's trunk and started messing with him.
The elephant started crying in pain. He understood his mistake and apologised to the ants
and every one of the animals he had harassed.
'''
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,coref')
    metric_val = average_mentions_per_coref_chain(story,nlp)
    print(f"Average Coreference Val: {metric_val}")


if __name__ == "__main__":
    main()
