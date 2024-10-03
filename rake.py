import nltk
from rake_nltk import Rake

# download only once - uncomment and run these 2 lines once and then comment them out
# nltk.download('stopwords')
# nltk.download('punkt_tab')


f = open("test.wp_source", "r")
text = f.read()

# Preprocessing
text = text.lower() 

# splitting corpus based on '.'
sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]


for line in sentences:


    stop_words = nltk.corpus.stopwords.words('english')
    r = Rake(stopwords=stop_words)
    keywords = r.extract_keywords_from_text(line)

    print(f"Sentence: {line}")
    keyword_lst = r.get_ranked_phrases()[:10]
    print(f"Keywords: {keyword_lst}")
    keyword_lst = r.get_ranked_phrases_with_scores()
    print("\n")
    print("Similarity Scores:")
    for elem in keyword_lst:
        print(f"{elem[1]} : {elem[0]}")
    
    print("\n---------------------------\n")