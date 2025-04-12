import nltk
from nltk.stem import PorterStemmer
import string

nltk.download("punkt")  # Download the Punkt tokenizer if not already installed

stemmer = PorterStemmer()


# Tokenize function
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Stem each word
def stem(word):
    return stemmer.stem(word.lower())  # Lowercase before stemming


# Bag of words for a sentence
def bag_of_words(sentence, all_words):
    sentence_words = [
        stem(w) for w in tokenize(sentence)
    ]  # Tokenize and stem each word in the sentence
    bag = [
        1 if w in sentence_words else 0 for w in all_words
    ]  # Create a bag of words array
    return bag
