import spacy

nlp = spacy.load("en_core_web_sm") # Load spaCy model
nlp.add_pipe("textrank") # Add TextRank to the pipeline
text = "Your text here"
doc = nlp(text)
# Access top phrases using doc._.phrases[:n] where n is the desired number
for phrase in doc._.phrases[:10]:
  print(phrase.text)