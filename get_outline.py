from rake_nltk import Rake
from summa import summarizer
from utils import get_nth_line

def generate_abstract(text, ratio=0.3):
    """
    Generates an abstract of the input text by keeping a percentage of the original text.
    
    :param text: str, The input text to summarize.
    :param ratio: float, The ratio of sentences to keep (e.g., 0.3 for 30%).
    :return: str, The summarized abstract text.
    """
    return summarizer.summarize(text, ratio=ratio)

def extract_keywords(text):
    """
    Extracts keywords from the input text using RAKE (Rapid Automatic Keyword Extraction) algorithm.
    
    :param text: str, The input text to extract keywords from.
    :return: list, A list of extracted keywords ordered by their relevance.
    """
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def generate_outline(text, summary_ratio=0.3):
    """
    Combines abstract and keywords to create a story outline.
    
    :param text: str, The input text to generate an outline from.
    :param summary_ratio: float, The ratio of text to keep in the abstract.
    :return: dict, An outline consisting of the abstract and key story elements.
    """
    abstract = generate_abstract(text, ratio=summary_ratio)
    keywords = extract_keywords(text)
    print("\nKeywords: ", keywords)
    # # Create an outline
    # outline = {
    #     "Abstract (Story Plot)": abstract,
    #     "Key Elements (Themes/Concepts)": keywords[:10],  # Top 5 keywords for simplicity
    #     "Detailed Story Ideas": [
    #         f"Expand on: {keyword}" for keyword in keywords[:10]
    #     ]  # Prompts to expand the keywords into story ideas
    # }
    
    outline = [abstract] + keywords[:10]
    
    return outline

file_path = "archive/writingPrompts/train.wp_target"
n = 1

prompt = get_nth_line("archive/writingPrompts/train.wp_source", n)
print("Prompt: ", prompt)

text = get_nth_line(file_path, n)
print("Story: ", text)

if text:
    story_outline = generate_outline(text, summary_ratio=0.3)
    print("Story Outline: ", story_outline)
else:
    print(f"Line {n} doesn't exist in the file.")
