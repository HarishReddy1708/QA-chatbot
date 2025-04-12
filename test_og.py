import json
from utils.nltk_utils import stem
import torch
from src.model import NeuralNet
from utils.utils_og import bag_of_words, tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import string
import numpy as np
import os


import random
import re

with open("data/intents.json", "r") as f:
    intents = json.load(f)

# Load the trained model data
FILE = "weights\chatdata1000.pth"  # Path to the trained model file
data = torch.load(FILE)

# Hyperparameters (from training phase)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(data["model_state"])
model.eval()  # Set the model to evaluation mode


# Function to predict the intent of a sentence
def predict_intent(sentence):
    # Tokenize and stem the input sentence
    tokens = tokenize(sentence)
    tokens = [stem(w) for w in tokens]

    # Create the bag of words
    X = bag_of_words(tokens, all_words)
    X = np.array(X)
    X = torch.from_numpy(X).float()

    # Get the model's output
    output = model(X)

    # Get the predicted tag (class index with highest probability)
    _, predicted_idx = torch.max(output, dim=0)
    predicted_tag = tags[predicted_idx.item()]

    return predicted_tag


# Sample test questions from the intents (1–2 per intent works well)
test_samples = [
    # Greeting Intent
    ("Hello!", "greeting"),
    ("Hey there!", "greeting"),
    ("How’s it going?", "greeting"),
    ("Good morning!", "greeting"),
    ("What’s up?", "greeting"),
    ("Hi!", "greeting"),
    ("Yo!", "greeting"),
    ("Hey!", "greeting"),
    # Goodbye Intent
    ("See you later!", "goodbye"),
    ("Take care!", "goodbye"),
    ("Thanks for your help!", "goodbye"),
    ("Goodbye!", "goodbye"),
    ("Have a good day!", "goodbye"),
    ("Catch you later!", "goodbye"),
    ("Farewell!", "goodbye"),
    ("See ya!", "goodbye"),
    # Thanks Intent
    ("Thank you so much!", "thanks"),
    ("I appreciate it!", "thanks"),
    ("Thanks a lot!", "thanks"),
    ("Much appreciated!", "thanks"),
    ("Thanks for everything!", "thanks"),
    ("Thanks a bunch!", "thanks"),
    ("Thank you very much!", "thanks"),
    # Test Drive Intent
    ("How can I schedule a test drive?", "test_drive"),
    ("Can I test drive the Porsche?", "test_drive"),
    ("What are the test drive options?", "test_drive"),
    ("I want to book a test drive.", "test_drive"),
    ("How do I arrange a test drive?", "test_drive"),
    ("When can I schedule a test drive?", "test_drive"),
    ("Is it possible to test drive a Porsche?", "test_drive"),
    # Subjective Questions Intent
    ("What’s the best Porsche to drive for fun?", "subjective_questions"),
    ("Which Porsche model has the best handling?", "subjective_questions"),
    ("Which Porsche is the fastest?", "subjective_questions"),
    ("Which Porsche has the best design?", "subjective_questions"),
    ("What’s the most fun Porsche to drive?", "subjective_questions"),
    ("Which Porsche is the most powerful?", "subjective_questions"),
    # Model Comparison Intent
    ("How do the Porsche models compare?", "model_comparison"),
    ("What is the difference between the Cayenne and the Macan?", "model_comparison"),
    ("Which Porsche SUV is better?", "model_comparison"),
    ("Can you compare the Taycan and the 911?", "model_comparison"),
    ("Which Porsche is better for off-road driving?", "model_comparison"),
    ("How does the Taycan compare to the 911?", "model_comparison"),
    # Vibe: Classic or Cutting Edge Intent
    ("I want something classic.", "vibe_classic_cutting_edge"),
    ("I prefer futuristic designs.", "vibe_classic_cutting_edge"),
    (
        "What’s the difference between the 911 and the Taycan?",
        "vibe_classic_cutting_edge",
    ),
    ("Which Porsche has a more vintage feel?", "vibe_classic_cutting_edge"),
    (
        "Do you have any Porsche models with a modern aesthetic?",
        "vibe_classic_cutting_edge",
    ),
    ("What’s the most classic Porsche model?", "vibe_classic_cutting_edge"),
    # Preference: Gas or Electric Intent
    ("I like gas-powered engines.", "preference_gas_electric"),
    ("Tell me about electric Porsche cars.", "preference_gas_electric"),
    ("What’s better, gas or electric?", "preference_gas_electric"),
    ("Do you have electric Porsche models?", "preference_gas_electric"),
    ("Can I get an electric Porsche?", "preference_gas_electric"),
    ("Which Porsche is electric?", "preference_gas_electric"),
    # Timeless or Modern Looks Intent
    ("Which Porsche has a classic look?", "timeless_modern_looks"),
    ("Do you have modern-looking Porsches?", "timeless_modern_looks"),
    ("What’s more timeless, the 911 or the Taycan?", "timeless_modern_looks"),
    ("I want a car with a sleek, modern design.", "timeless_modern_looks"),
    ("Which Porsche is more timeless in design?", "timeless_modern_looks"),
    ("What Porsche has the most modern aesthetic?", "timeless_modern_looks"),
    # Heritage or High-Tech Intent
    ("Which Porsche is more traditional?", "heritage_high_tech"),
    ("What’s better, the 911 or the Taycan?", "heritage_high_tech"),
    ("Do you have a Porsche that’s more about legacy?", "heritage_high_tech"),
    ("What is the Porsche of the future?", "heritage_high_tech"),
    ("Is the 911 more traditional than the Taycan?", "heritage_high_tech"),
    ("Which Porsche embraces more advanced technology?", "heritage_high_tech"),
    # Driving Style Preference Intent
    ("Which Porsche is better for speed?", "driving_style_preference"),
    ("I need a comfortable Porsche.", "driving_style_preference"),
    (
        "What Porsche should I drive for a thrilling experience?",
        "driving_style_preference",
    ),
    ("Which Porsche gives the best performance?", "driving_style_preference"),
    ("Which Porsche is better for a road trip?", "driving_style_preference"),
    ("Which Porsche is great for daily commuting?", "driving_style_preference"),
    # Color Preference Intent
    ("What are the boldest Porsche colors?", "color_preference"),
    ("What’s the most classic Porsche color?", "color_preference"),
    ("Which Porsche comes in Miami Blue?", "color_preference"),
    ("Can I get a Porsche in Racing Yellow?", "color_preference"),
    ("Which Porsche has the best paint options?", "color_preference"),
    ("What color options do you have for the 911?", "color_preference"),
    # Model Experience Intent
    ("What’s the best Porsche model?", "model_experience"),
    ("Which Porsche should I choose for speed?", "model_experience"),
    ("Tell me about the ideal Porsche model.", "model_experience"),
    ("What’s the perfect Porsche for me?", "model_experience"),
    ("Which Porsche suits my lifestyle?", "model_experience"),
    ("What’s the ultimate Porsche experience?", "model_experience"),
    # Customization Options Intent
    ("Can I customize my Porsche?", "customization_options"),
    ("How can I build my own Porsche?", "customization_options"),
    ("What customization options are available?", "customization_options"),
    ("Tell me about the Porsche configurator.", "customization_options"),
    ("Can I choose my Porsche interior color?", "customization_options"),
    ("How do I personalize my Porsche?", "customization_options"),
    # Fastest Model Intent
    ("Which is the fastest 911 model?", "fastest_model"),
    ("Which Porsche has the highest top speed?", "fastest_model"),
    ("Tell me about the fastest Porsche.", "fastest_model"),
    ("What is the top speed of the Taycan?", "fastest_model"),
    ("Which Porsche has the best acceleration?", "fastest_model"),
    ("What’s the fastest Porsche you offer?", "fastest_model"),
    # Most Torque Intent
    ("Which Porsche has the most torque?", "most_torque"),
    ("Tell me about the 911 with the highest torque.", "most_torque"),
    ("Which model has the highest torque?", "most_torque"),
    ("What’s the torque of the Cayenne Turbo?", "most_torque"),
    ("Which Porsche has the strongest torque?", "most_torque"),
    ("Which Porsche offers the best torque output?", "most_torque"),
    # Cheapest Model Intent
    ("What’s the cheapest Porsche 911?", "cheapest_model"),
    ("Which Porsche is the most affordable?", "cheapest_model"),
    ("Tell me about the entry-level Porsche.", "cheapest_model"),
    ("What’s the cheapest model of Porsche?", "cheapest_model"),
    ("Which Porsche is the least expensive?", "cheapest_model"),
    ("What’s the price of the cheapest 911?", "cheapest_model"),
    # Best Output per Liter Intent
    ("Which Porsche has the best power per liter?", "best_output_per_liter"),
    ("Which model offers the best engine efficiency?", "best_output_per_liter"),
    (
        "Tell me about the Porsche with the highest output per liter.",
        "best_output_per_liter",
    ),
    ("Which Porsche has the best engine performance?", "best_output_per_liter"),
    ("What’s the most efficient Porsche engine?", "best_output_per_liter"),
    ("Which Porsche delivers the most power for the size?", "best_output_per_liter"),
    # Most Color Options Intent
    ("Which Porsche has the most color options?", "most_color_options"),
    ("What Porsche offers the best color choices?", "most_color_options"),
    ("Which 911 model comes in the most colors?", "most_color_options"),
    ("How many color choices are available for the Porsche?", "most_color_options"),
    ("Can I get a Porsche in different colors?", "most_color_options"),
    ("Which Porsche offers the widest color palette?", "most_color_options"),
    # Model Spec Query Intent
    ("What is the horsepower of the 911 Carrera?", "model_spec_query"),
    ("How much torque does the Cayenne Turbo have?", "model_spec_query"),
    ("What’s the top speed of the Taycan?", "model_spec_query"),
    ("Which Porsche has the highest output?", "model_spec_query"),
    ("How powerful is the 911 Turbo?", "model_spec_query"),
    ("What’s the horsepower of the Taycan?", "model_spec_query"),
]

# Evaluate accuracy
true_labels = []
predictions = []

for sentence, actual_tag in test_samples:
    predicted_label = predict_intent(sentence)

    true_labels.append(actual_tag)
    predictions.append(predicted_label)

    print(
        f"Input: '{sentence}' | Predicted: '{predicted_label}' | Expected: '{actual_tag}' | {'✅' if predicted_label == actual_tag else '❌'}"
    )

accuracy = np.mean(np.array(true_labels) == np.array(predictions)) * 100
print(f"\nAccuracy on {len(test_samples)} test samples: {accuracy:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(true_labels, predictions, labels=tags)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tags, yticklabels=tags)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report for precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=tags))
