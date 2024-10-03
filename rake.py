import nltk
from rake_nltk import Rake
# nltk.download('stopwords')

# Sample text
f = open("train.wp_source", "r")
# text = "Natural Language Processing (NLP) is a field of Artificial Intelligence concerned with enabling computers to understand and process human language. It is used to apply machine learning algorithms to text and speech. NLP is used in a variety of applications, such as language translation, sentiment analysis, and chatbots."
text = f.read()

# Preprocess (replace with your cleaning steps)
text = text.lower() # Lowercase for case-insensitive stop word removal

# NLTK stop words
stop_words = nltk.corpus.stopwords.words('english')

# RAKE initialization
r = Rake(stopwords=stop_words)

# Extract keywords
keywords = r.extract_keywords_from_text(text)

# Print top 3 keywords
print(r.get_ranked_phrases()[:10])