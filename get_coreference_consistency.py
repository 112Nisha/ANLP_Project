import stanza

def get_coreference_clusters(prompt):
    # stanza.download('en')
    nlp = stanza.Pipeline('en')
    return nlp(prompt)