import argparse
import os

from summa.summarizer import summarize
from summa.keywords import keywords

# Types of summarization
SENTENCE = 0
WORD = 1

DEFAULT_RATIO = 0.3

def get_test_file_path(file):
    pre_path = os.path.join(os.path.dirname(__file__), 'test_data')
    return os.path.join(pre_path, file)

def get_text_from_test_data(file):
    file_path = get_test_file_path(file)
    with open(file_path, mode='r', encoding="utf-8") as f:
        return f.read()
    
def textrank(text, summarize_by=SENTENCE, ratio=DEFAULT_RATIO, words=None, additional_stopwords=None):
    if summarize_by == SENTENCE:
        return summarize(text, ratio, words, additional_stopwords=additional_stopwords)
    else:
        return keywords(text, ratio, words, additional_stopwords=additional_stopwords)

def existing_file(file_name):
    try:
        with open(file_name, 'r') as file:
            return file.read()
    except Exception:
        raise argparse.ArgumentTypeError("The file provided could not be opened.")

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("{} not in range [0.0, 1.0]".format(x))
    return x

def parse_args(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, prog="textrank", description="Extract the most relevant sentences or keywords of a given text using the TextRank algorithm.")

    group = parser.add_mutually_exclusive_group(required=True)
    # New API
    group.add_argument('--summarize', metavar="path/to/file", type=existing_file,
                       help="Run textrank to summarize the input text.")
    group.add_argument('--keywords', metavar="path/to/file", type=existing_file,
                       help="Run textrank to extract keywords from the input text.")
    # Old API
    group.add_argument('--text', '-t', metavar="path/to/file", type=existing_file,
                       help="(Deprecated) Text to summarize if --summary option is selected")

    parser.add_argument('--summary', '-s', metavar="{0,1}", type=int, choices=[SENTENCE, WORD], default=0,
                        help="(Deprecated) Type of unit to summarize: sentence (0) or word (1)")
    parser.add_argument('--ratio', '-r', metavar="r", type=restricted_float, default=DEFAULT_RATIO,
                        help="Float number (0,1] that defines the length of the summary. It's a proportion of the original text")
    parser.add_argument('--words', '-w', metavar="#words", type=int,
                        help="Number to limit the length of the summary. The length option is ignored if the word limit is set.")
    parser.add_argument('--additional_stopwords', '-a', metavar="list,of,stopwords",
                        help="Either a string of comma separated stopwords or a path to a file which has comma separated stopwords in every line")

    return parser.parse_args(args)

def textrank_for_story(text):
    mode = SENTENCE
    ratio = DEFAULT_RATIO
    additional_stopwords = None  
   
    # Make a summary of the text
    words = len(text.split()) * DEFAULT_RATIO  # Number of words in the summary
    generated_summary = textrank(text, mode, ratio=ratio, words=words, additional_stopwords=additional_stopwords)  # Generate summary
    return generated_summary # Return the summary

def main():
    """
    Main function to run textrank as an example for story generation
    """

    text = """I'm someone who loves learning and exploring new ideas. Whether it's understanding 
    how things work or diving into books and projects, I enjoy the challenge of figuring things out. 
    I like spending time working on creative tasks, whether it's solving problems with code or thinking 
    about big ideas. When I'm not busy with work or school, I enjoy relaxing by listening to music, 
    reading, or going for walks to clear my mind. I'm also a fan of having deep conversations about life, 
    society, and what makes people tick. Overall, I enjoy balancing hard work with moments of peace and 
    reflection."""

    print("Summarized text:")
    print(textrank_for_story(text))

if __name__ == "__main__":
    main()
