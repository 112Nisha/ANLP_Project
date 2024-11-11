from rake_nltk import Rake
from summa import summarizer

def generate_abstract(text, ratio=0.3):
    return summarizer.summarize(text, ratio=ratio)

def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def generate_outline(text, summary_ratio=0.3):
    abstract = generate_abstract(text, ratio=summary_ratio)
    keywords = extract_keywords(text)
    outline = [abstract] + keywords[:10]
    return outline
