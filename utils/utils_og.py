import nltk
import numpy as np
import random
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download("punkt")  # To ensure that word_tokenize is available

# Initialize the stemmer
stemmer = PorterStemmer()


# Tokenizer function to split sentence into words
def tokenize(sentence):
    return word_tokenize(sentence.lower())


# Function to stem a word
def stem(word):
    return stemmer.stem(word.lower())


# Function to create a bag of words (for input representation)
def bag_of_words(tokens, all_words):
    """
    Convert a list of tokenized words into a bag of words,
    represented as a list where 1 means the word is present and 0 means it's absent.
    """
    tokens = [stem(w) for w in tokens]  # Stem the words
    bag = np.zeros(len(all_words), dtype=int)  # Initialize the bag as zeros

    # For each word in the tokenized input, mark its presence in the bag
    for idx, word in enumerate(all_words):
        if word in tokens:
            bag[idx] = 1

    return bag


# Function to handle cleaning and preprocessing of text
def clean_sentence(sentence):
    """
    Remove punctuation and split the sentence into words.
    This will return the cleaned sentence with only the relevant words.
    """
    sentence = sentence.translate(
        str.maketrans("", "", string.punctuation)
    )  # Remove punctuation
    words = word_tokenize(sentence.lower())  # Tokenize and convert to lowercase
    return words
