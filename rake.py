import nltk
from rake_nltk import Rake

# download only once - uncomment and run these 2 lines once and then comment them out
# nltk.download('stopwords')
# nltk.download('punkt_tab')


# f = open("test.wp_source", "r")
# text = f.read()

text = """
Once in a small village, nestled between rolling hills, there lived a kind-hearted baker named Elara. She was known for her delicious breads and pastries, but even more so for her generous spirit. Every morning, she would wake up before dawn, kneading dough and crafting treats, which she would share with anyone in need.

One chilly winter’s morning, as she opened her bakery, she noticed an elderly man sitting outside, shivering and alone. His clothes were tattered, and he looked weary from wandering. Without hesitation, Elara brought him a warm loaf of bread and a steaming cup of tea.

“Thank you, dear lady,” he said, his voice trembling. “I haven’t eaten in days.”

Elara smiled, her heart swelling with warmth. “You are welcome to stay here as long as you need. We have more than enough bread to share.”

As the days turned into weeks, the old man, whom Elara learned was named Finn, became a part of the bakery’s daily life. He helped her with chores, and in return, Elara taught him the art of baking. The villagers noticed the transformation in Finn; his eyes sparkled with joy as he created delightful pastries alongside Elara.

One day, a heavy snowstorm struck the village, isolating it from the outside world. The villagers gathered in Elara's bakery, seeking refuge from the cold. She welcomed them all, serving warm bread and sharing stories by the fire. Finn, inspired by Elara’s kindness, suggested they bake a large batch of bread to share with those who couldn’t make it to the bakery.

With everyone’s help, they baked hundreds of loaves, which Elara and Finn delivered to homes around the village. The warmth of the fresh bread spread through the streets, melting the cold and filling hearts with hope.

When the storm passed, the villagers came together to celebrate their unity. They named the annual gathering “The Festival of Bread,” a tribute to Elara and Finn’s spirit of generosity. From then on, every winter, they would bake together, reminding one another of the power of kindness, community, and the simple joy of sharing a meal.

And so, Elara’s bakery flourished, not just as a place of delicious treats, but as a symbol of love and togetherness in the heart of the village.
"""

# Preprocessing
text = text.lower() 




stop_words = nltk.corpus.stopwords.words('english')
r = Rake(stopwords=stop_words)
keywords = r.extract_keywords_from_text(text)

print(f"Text: {text}")
keyword_lst = r.get_ranked_phrases()[:10]
print(f"Keywords: {keyword_lst}")
keyword_lst = r.get_ranked_phrases_with_scores()
print("\n")
print("Similarity Scores:")
for elem in keyword_lst:
    print(f"{elem[1]} : {elem[0]}")

print("\n---------------------------\n")
