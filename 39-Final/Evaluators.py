import torch
from params import THRESHOLD
from CorefClusters import get_coref_clusters
from GoldenBert import predict_discourse_marker
import logging
logging.getLogger("stanza").setLevel(logging.ERROR) 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

BATCH_SIZE = 8

def calculate_perplexity(loss_value):
    return torch.exp(torch.tensor(loss_value))

def calculate_distinct_1_2(generated_text):
    total_1_grams = []
    total_2_grams = []
    
    words = generated_text.split()
    total_1_grams.extend(words)
    for index in range(len(words) - 1):
        total_2_grams.append((words[index], words[index + 1]))
    unique_1_grams = set(total_1_grams)
    unique_2_grams = set(total_2_grams)
    if len(total_1_grams) == 0:
        distinct_1 = 0
    else:
        distinct_1 = len(unique_1_grams) / (len(total_1_grams)) 
    if len(total_2_grams) == 0:
        distinct_2 = 0
    else:
        distinct_2 = len(unique_2_grams) / (len(total_2_grams))      
    return distinct_1, distinct_2

def evaluate_coref_coherence(story, nlp):
    clusters = get_coref_clusters(story, nlp)
    if clusters == -1:
        # print(f"Story ERR!! {story}")
        return 0
    num_chains = len(clusters)
    if num_chains == 0:
        return 0
    total_mentions = sum(len(cluster["mentions"]) for cluster in clusters)
    return total_mentions / num_chains if num_chains > 0 else 0

def coherence_score(generated_text, golden_bert, golden_bert_tokenizer, device):
    generated_text = generated_text.split(".!?")
    num_unknown = 0
    total = len(generated_text)

    for i in range(total-1):
        marker, confidence = predict_discourse_marker(golden_bert, golden_bert_tokenizer, generated_text[i], generated_text[i+1], device)
        # print(marker, confidence)
        if confidence < THRESHOLD:
            num_unknown += 1
    return num_unknown / total
