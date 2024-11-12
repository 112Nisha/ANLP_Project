import torch
import stanza 
import logging
import warnings
logging.getLogger("stanza").setLevel(logging.ERROR) 
warnings.filterwarnings("ignore", category=FutureWarning)

def get_coref_clusters(story, nlp):
    print(story)
    doc = nlp(story)
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

story = '''
There was once a pleased elephant who generally harassed smaller animals.
He would go to the ant colony and shower water on the ants.
The ants, with their size, could just cry.
The elephant laughed and threatened the ants that he would kill them.
The ants had enough and chose to show the elephant a lesson.
They went straight into the elephant's trunk and started messing with him.
The elephant started crying in pain. He understood his mistake and apologised to the ants
and every one of the animals he had harassed.
'''

# attention_weights = torch.rand(10, 10)  
# loss = coreference_loss(attention_weights, story, nlp)
# print(f"Coreference Loss: {loss}")
