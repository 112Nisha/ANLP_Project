import nltk
from rake_nltk import Rake

# Download stopwords and punkt, if needed by uncommenting the following lines:
# nltk.download('stopwords')
# nltk.download('punkt_tab')

def get_keywords(text, n):
    """
    Function to extract keywords from text using Rake algorithm
    Arguments:
        text : str : input text
        n : int : number of keywords to extract
    Returns:
        keyword_lst : list : list of keywords
    """
    text = text.lower()  # convert to lowercase
    stop_words = nltk.corpus.stopwords.words('english')  # get stopwords
    r = Rake(stopwords=stop_words)  # initialize Rake with stopwords
    keyword_lst = r.get_ranked_phrases()[:n]  # get keywords

    # Uncomment the following lines to get keywords with scores
    # keyword_lst = r.get_ranked_phrases_with_scores() # get keywords with scores
    # print("Similarity Scores:")
    # for elem in keyword_lst:
    #     print(f"{elem[1]} : {elem[0]}")
    return keyword_lst
