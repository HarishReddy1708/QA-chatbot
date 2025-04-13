import nltk
from nltk.stem.porter import PorterStemmer
import json
import numpy as np
from nltk.tokenize import word_tokenize
import string
import os

# Ensure NLTK data is downloaded
""" try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt") """

nltk_data_path = os.environ.get("NLTK_DATA", "/usr/share/nltk_data")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)
stemmer = PorterStemmer()

# Load the car data from the updated cars.json file to recognize car models
try:
    with open("cars.json") as f:
        car_data = json.load(f)
except FileNotFoundError:
    print("Error: cars.json file not found!")
    car_data = {"models": []}

# Extract car names from the "models" list
car_names = [car["name"] for car in car_data["models"]]

# This will now contain all car names
# print("Car Names:", car_names)


def tokenize(sentence):
    """
    Tokenize sentence and check for car names in the sentence
    """
    try:
        # Tokenize the sentence using word_tokenize from nltk
        words = word_tokenize(sentence)

        # Remove punctuation from the tokenized words
        words = [word.lower() for word in words if word not in string.punctuation]

        # Check for car names in the filtered words
        recognized_cars = [
            car for car in car_names if car.lower() in " ".join(words).lower()
        ]

        # Return both the tokenized sentence and recognized cars
        return words, recognized_cars
    except Exception as e:
        print(f"Error in tokenization: {str(e)}")
        return [], []


def stem(word):
    """
    Stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    try:
        return stemmer.stem(word.lower())  # Stem each word
    except Exception as e:
        print(f"Error in stemming: {str(e)}")
        return word.lower()


def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    try:
        # Stem each word
        sentence_words = [stem(word) for word in tokenized_sentence]
        # Initialize bag with 0 for each word
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1
        return bag
    except Exception as e:
        print(f"Error in bag_of_words: {str(e)}")
        return np.zeros(len(words), dtype=np.float32)


# Function to extract car features when a car is recognized
def get_car_features(car_name):
    """
    Get features of a recognized car model
    """
    try:
        for car in car_data["models"]:
            if car["name"].lower() == car_name.lower():
                return car["features"]
        return None
    except Exception as e:
        print(f"Error getting car features: {str(e)}")
        return None


# Testing tokenize function with a test sentence
sentence = "Tell me about the Porsche 911 Carrera."
tokenized_sentence, recognized_cars = tokenize(sentence)

# Output the tokenized words and recognized cars
# print("Tokenized Sentence:", tokenized_sentence)
# print("Recognized Cars:", recognized_cars)

# If any car was recognized, print its features
for car in recognized_cars:
    features = get_car_features(car)
    # print(f"Features of {car}: {features}")
