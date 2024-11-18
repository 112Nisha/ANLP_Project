# !pip install stanza
# !pip install peft
# !pip install rake_nltk
# !pip install summa
# !pip install nltk
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

import torch
import stanza 
import logging
import warnings
logging.getLogger("stanza").setLevel(logging.ERROR) 
warnings.filterwarnings("ignore", category=FutureWarning)

def get_coref_clusters(story, nlp):
   #  print(story)
    try:
        doc = nlp(story)
        
    except:
        return -1
    if doc.coref == []:
        return -1
    clusters = []
    for i, coref_chain in enumerate(doc.coref):
        cluster_info = {
            "cluster_id": i + 1,
            "mentions": []
        }

        for mention in coref_chain.mentions:
            start_idx = mention.start_word
            end_idx = mention.end_word
            mention_text = " ".join(
                [word.text for sentence in doc.sentences for word in sentence.words 
                 if start_idx <= word.id <= end_idx]
            )
            cluster_info["mentions"].append({
                "mention_text": mention_text,
                "start_index": start_idx,
                "end_index": end_idx
            })

        clusters.append(cluster_info)
    return clusters

def coreference_loss(attention_weights, story, nlp):
    coref_clusters = get_coref_clusters(story, nlp)
    if coref_clusters == -1:
        return 0.0
    if not coref_clusters:  # If empty list or None
        return torch.tensor(0.0, device=attention_weights.device)
    num_clusters = len(coref_clusters)
    total_mentions = sum(len(cluster["mentions"]) for cluster in coref_clusters)
    loss = 0

    for cluster in coref_clusters:
        mentions = cluster["mentions"]
        num_mentions = len(mentions)

        if num_mentions > 1:
            cluster_loss = 0

            for i in range(num_mentions):
                for j in range(i + 1, num_mentions):
                    start_idx_1 = mentions[i]["start_index"]
                    start_idx_2 = mentions[j]["start_index"]

                    idx_1 = start_idx_1 % attention_weights.size(0)
                    idx_2 = start_idx_2 % attention_weights.size(1)

                    alpha_ik = attention_weights[idx_1, idx_2]

                    if alpha_ik > 0:
                        cluster_loss -= torch.log(alpha_ik)

            # Scale by 1/N_i for this cluster
            cluster_loss /= num_mentions
            loss += cluster_loss

    # Scale by -(1 / (p * q))
    if num_clusters * total_mentions > 0:
        loss = -(1 / (num_clusters * total_mentions)) * loss
    return loss

nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,coref')
